import { useRef, useEffect, useCallback } from "react";
import * as topojson from "topojson-client";
import type { Topology } from "topojson-specification";

// ── Types ───────────────────────────────────────────────────────────────────

interface DataWorldMapProps {
  isLoading: boolean;
  hasData: boolean;
  ticker: string | null;
}

interface CityNode {
  label: string;
  lat: number;
  lon: number;
  primary?: boolean;
}

interface Particle {
  fromIdx: number;
  toIdx: number;
  t: number;
  speed: number;
  size: number;
  opacity: number;
}

type AnimState = "idle" | "loading" | "loaded";
type LandPolygon = number[][][]; // polygon[ring][point][lon,lat]

// ── Constants ───────────────────────────────────────────────────────────────

const ACCENT = "#00e5ff";
const USER_COLOR = "#ff3333";
const BG_R = 5, BG_G = 10, BG_B = 18; // #050a12 decomposed
const MAX_PARTICLES = 200;

const LAT_MAX = 90;
const LAT_MIN = -60;
const MAP_ASPECT = 360 / (LAT_MAX - LAT_MIN); // ~2.4

const TOPO_URL = "https://cdn.jsdelivr.net/npm/world-atlas@2/land-50m.json";

// ── Data source cities ──────────────────────────────────────────────────────

const DATA_NODES: CityNode[] = [
  { label: "SFO", lat: 37.77, lon: -122.42 },
  { label: "CHI", lat: 41.88, lon: -87.63 },
  { label: "TOR", lat: 43.65, lon: -79.38 },
  { label: "NYC", lat: 40.71, lon: -74.01, primary: true },
  { label: "LDN", lat: 51.51, lon: -0.13 },
  { label: "FRA", lat: 50.11, lon: 8.68 },
  { label: "MUM", lat: 19.08, lon: 72.88 },
  { label: "SHA", lat: 31.23, lon: 121.47 },
  { label: "HKG", lat: 22.32, lon: 114.17 },
  { label: "TKY", lat: 35.68, lon: 139.69 },
  { label: "SYD", lat: -33.87, lon: 151.21 },
  { label: "SAO", lat: -23.55, lon: -46.63 },
];

const USER_NODE_IDX = 12;

// ── Projection ──────────────────────────────────────────────────────────────

interface Viewport {
  ox: number; // offsetX
  oy: number; // offsetY
  mw: number; // map width
  mh: number; // map height
}

function computeViewport(sw: number, sh: number): Viewport {
  const aspect = sw / sh;
  let mw: number, mh: number, ox: number, oy: number;
  if (aspect > MAP_ASPECT) {
    mh = sh;
    mw = mh * MAP_ASPECT;
    ox = (sw - mw) / 2;
    oy = 0;
  } else {
    mw = sw;
    mh = mw / MAP_ASPECT;
    ox = 0;
    oy = (sh - mh) / 2;
  }
  return { ox, oy, mw, mh };
}

/** lon → screen X */
function projX(lon: number, vp: Viewport): number {
  return vp.ox + ((lon + 180) / 360) * vp.mw;
}

/** lat → screen Y (clamped to LAT_MIN..LAT_MAX for nodes/particles) */
function projY(lat: number, vp: Viewport): number {
  const c = Math.max(LAT_MIN, Math.min(LAT_MAX, lat));
  return vp.oy + ((LAT_MAX - c) / (LAT_MAX - LAT_MIN)) * vp.mh;
}

/** lat → screen Y (unclamped — for polygon fills so boundary points project beyond edge) */
function projYRaw(lat: number, vp: Viewport): number {
  return vp.oy + ((LAT_MAX - lat) / (LAT_MAX - LAT_MIN)) * vp.mh;
}

// ── Bezier arc helper ───────────────────────────────────────────────────────

function bezierPoint(
  x1: number, y1: number,
  x2: number, y2: number,
  t: number,
  mapW: number,
): { x: number; y: number } {
  const mx = (x1 + x2) / 2;
  const my = (y1 + y2) / 2;
  const bulge = Math.min(Math.abs(x2 - x1) * 0.35, mapW * 0.12);
  const cx = mx;
  const cy = my - bulge;
  const u = 1 - t;
  return {
    x: u * u * x1 + 2 * u * t * cx + t * t * x2,
    y: u * u * y1 + 2 * u * t * cy + t * t * y2,
  };
}

// ── Component ───────────────────────────────────────────────────────────────

declare global {
  interface Window {
    __mapState?: AnimState;
  }
}

export default function DataWorldMap({ isLoading, hasData, ticker }: DataWorldMapProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rafRef = useRef(0);
  const stateRef = useRef<AnimState>("idle");
  const prevRef = useRef<AnimState>("idle");
  const burstTimeRef = useRef(0);
  const particlesRef = useRef<Particle[]>([]);
  const userLocRef = useRef({ lat: 40.71, lon: -74.01 });
  const nodesRef = useRef<CityNode[]>([]);
  const connsRef = useRef<[number, number][]>([]);
  const landRef = useRef<LandPolygon[]>([]);

  const resolveState = useCallback((): AnimState => {
    if (isLoading) return "loading";
    if (hasData && ticker) return "loaded";
    return "idle";
  }, [isLoading, hasData, ticker]);

  // ── Fetch TopoJSON once ─────────────────────────────────────────────────
  useEffect(() => {
    let dead = false;
    fetch(TOPO_URL)
      .then((r) => r.json())
      .then((topo: Topology) => {
        if (dead) return;
        const land = topojson.feature(topo, topo.objects.land) as GeoJSON.FeatureCollection;
        const polys: LandPolygon[] = [];
        for (const feat of land.features) {
          const g = feat.geometry;
          if (g.type === "Polygon") polys.push(g.coordinates);
          else if (g.type === "MultiPolygon") {
            for (const p of g.coordinates) polys.push(p);
          }
        }
        landRef.current = polys;
      })
      .catch(() => {});
    return () => { dead = true; };
  }, []);

  // ── Canvas + animation ──────────────────────────────────────────────────
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let alive = true;
    let lastT = 0;

    // -- Build node list & connections --
    function rebuild() {
      const user: CityNode = {
        label: "YOU",
        lat: userLocRef.current.lat,
        lon: userLocRef.current.lon,
        primary: true,
      };
      nodesRef.current = [...DATA_NODES, user];
      // Every city connects to user
      const c: [number, number][] = [];
      for (let i = 0; i < DATA_NODES.length; i++) {
        c.push([i, USER_NODE_IDX]);
      }
      connsRef.current = c;
    }

    // Geolocation with fallback
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (pos) => {
          userLocRef.current = { lat: pos.coords.latitude, lon: pos.coords.longitude };
          rebuild();
        },
        () => rebuild(),
      );
    }
    rebuild();

    // -- Resize --
    function resize() {
      const dpr = window.devicePixelRatio || 1;
      const w = window.innerWidth;
      const h = window.innerHeight;
      canvas!.width = w * dpr;
      canvas!.height = h * dpr;
      canvas!.style.width = w + "px";
      canvas!.style.height = h + "px";
      ctx!.setTransform(dpr, 0, 0, dpr, 0, 0);
    }
    resize();
    window.addEventListener("resize", resize);

    // -- Particle spawner --
    function spawn(): Particle {
      const cs = connsRef.current;
      const ci = Math.floor(Math.random() * cs.length);
      const [from, to] = cs[ci];
      return {
        fromIdx: from,
        toIdx: to,
        t: Math.random(),
        speed: 0.02 + Math.random() * 0.03,
        size: 2 + Math.random() * 2.5,
        opacity: 0.4 + Math.random() * 0.5,
      };
    }

    // Seed particles
    if (particlesRef.current.length === 0) {
      for (let i = 0; i < 40; i++) particlesRef.current.push(spawn());
    }

    // -- Frame --
    function frame(now: number) {
      if (!alive) return;
      const dt = lastT ? (now - lastT) / 1000 : 0.016;
      lastT = now;

      const w = window.innerWidth;
      const h = window.innerHeight;
      const vp = computeViewport(w, h);
      const nodes = nodesRef.current;
      const particles = particlesRef.current;

      // State transitions
      const next = resolveState();
      if (next === "loaded" && prevRef.current === "loading") {
        burstTimeRef.current = now;
      }
      prevRef.current = stateRef.current;
      stateRef.current = next;
      window.__mapState = next;

      const state = next;
      const burstAge = (now - burstTimeRef.current) / 1000;
      const inBurst = burstAge < 1.5;

      // Dynamics per state
      let speedMul = 1;
      let targetCount = 40;
      let glowMul = 1;
      let pulseMul = 1;

      if (state === "loading") {
        speedMul = 4;
        targetCount = 150;
        glowMul = 2;
        pulseMul = 3;
      } else if (state === "loaded") {
        speedMul = inBurst ? 6 : 1.8;
        targetCount = inBurst ? 180 : 60;
        glowMul = inBurst ? 2.5 : 1.3;
        pulseMul = inBurst ? 4 : 1.5;
      }

      // Manage particle pool
      while (particles.length < targetCount && particles.length < MAX_PARTICLES) {
        particles.push(spawn());
      }
      while (particles.length > targetCount + 10) {
        particles.splice(Math.floor(Math.random() * particles.length), 1);
      }

      // ── Clear ──────────────────────────────────────────────────────
      ctx!.clearRect(0, 0, w, h);

      // ── Draw continent fills ───────────────────────────────────────
      const polys = landRef.current;
      if (polys.length > 0) {
        ctx!.fillStyle = "rgba(0, 229, 255, 0.08)";
        for (const poly of polys) {
          const ring = poly[0];
          if (!ring || ring.length < 3) continue;
          // Skip Antarctica
          if (ring.some((c) => c[1] < -70)) continue;

          ctx!.beginPath();
          for (let i = 0; i < ring.length; i++) {
            const px = projX(ring[i][0], vp);
            const py = projYRaw(ring[i][1], vp); // unclamped — boundary goes offscreen
            if (i === 0) ctx!.moveTo(px, py);
            else ctx!.lineTo(px, py);
          }
          ctx!.closePath();
          ctx!.fill();
        }
      }

      // ── Draw coastline strokes ─────────────────────────────────────
      if (polys.length > 0) {
        ctx!.strokeStyle = "rgba(0, 229, 255, 0.25)";
        ctx!.lineWidth = Math.max(0.8, Math.max(w, h) / 1200);

        for (const poly of polys) {
          const ring = poly[0];
          if (!ring || ring.length < 3) continue;
          if (ring.some((c) => c[1] < -70)) continue;

          ctx!.beginPath();
          let penDown = false;

          for (let i = 0; i < ring.length; i++) {
            const j = (i + 1) % ring.length;
            const lat1 = ring[i][1];
            const lat2 = ring[j][1];

            // Skip southern boundary segments
            if (lat1 < -55 || lat2 < -55) { penDown = false; continue; }

            // Skip flat segments at Natural Earth data boundary (~83.6°N).
            // These are artificial clipping lines, not real coastline.
            if (lat1 > 83 && lat2 > 83 && Math.abs(lat1 - lat2) < 1) {
              penDown = false;
              continue;
            }

            const x1 = projX(ring[i][0], vp);
            const y1 = projYRaw(lat1, vp);
            const x2 = projX(ring[j][0], vp);
            const y2 = projYRaw(lat2, vp);

            if (!penDown) { ctx!.moveTo(x1, y1); penDown = true; }
            ctx!.lineTo(x2, y2);
          }
          ctx!.stroke();
        }
      }

      // ── Particles (NO arc base lines drawn — particles only) ───────
      for (let i = particles.length - 1; i >= 0; i--) {
        const p = particles[i];
        p.t += p.speed * speedMul * dt;

        if (p.t >= 1) {
          particles[i] = spawn();
          particles[i].t = 0;
          continue;
        }

        if (p.fromIdx >= nodes.length || p.toIdx >= nodes.length) continue;
        const from = nodes[p.fromIdx];
        const to = nodes[p.toIdx];
        const x1 = projX(from.lon, vp);
        const y1 = projY(from.lat, vp);
        const x2 = projX(to.lon, vp);
        const y2 = projY(to.lat, vp);

        const pt = bezierPoint(x1, y1, x2, y2, p.t, vp.mw);

        // Glow
        const gr = (p.size + 3) * (state === "loading" ? 1.5 : 1);
        const gg = ctx!.createRadialGradient(pt.x, pt.y, 0, pt.x, pt.y, gr);
        gg.addColorStop(0, `rgba(0, 229, 255, ${p.opacity * 0.6})`);
        gg.addColorStop(1, "rgba(0, 229, 255, 0)");
        ctx!.fillStyle = gg;
        ctx!.fillRect(pt.x - gr, pt.y - gr, gr * 2, gr * 2);

        // Dot
        ctx!.beginPath();
        ctx!.arc(pt.x, pt.y, p.size * 0.6, 0, Math.PI * 2);
        ctx!.fillStyle = `rgba(0, 229, 255, ${p.opacity})`;
        ctx!.fill();
      }

      // ── Burst ripple on "loaded" ───────────────────────────────────
      if (inBurst && nodes.length > USER_NODE_IDX) {
        const un = nodes[USER_NODE_IDX];
        const cx = projX(un.lon, vp);
        const cy = projY(un.lat, vp);
        const radius = burstAge * 400;
        const alpha = Math.max(0, 0.3 * (1 - burstAge / 1.5));
        ctx!.beginPath();
        ctx!.arc(cx, cy, radius, 0, Math.PI * 2);
        ctx!.strokeStyle = `rgba(255, 51, 51, ${alpha})`;
        ctx!.lineWidth = 2;
        ctx!.stroke();
      }

      // ── City & user nodes ──────────────────────────────────────────
      const pulse = Math.sin(now * 0.001 * pulseMul) * 0.5 + 0.5;
      const scale = Math.max(w, h) / 1000;

      for (let n = 0; n < nodes.length; n++) {
        const nd = nodes[n];
        const nx = projX(nd.lon, vp);
        const ny = projY(nd.lat, vp);
        const isUser = n === USER_NODE_IDX;

        const baseR = (isUser ? 8 : nd.primary ? 6 : 4) * scale;
        const r = baseR + pulse * 3 * glowMul * scale;

        // Outer glow
        const glR = r * (isUser ? 5 : 3.5);
        const ga = (0.15 + pulse * 0.15) * glowMul;
        const grd = ctx!.createRadialGradient(nx, ny, 0, nx, ny, glR);
        if (isUser) {
          grd.addColorStop(0, `rgba(255, 51, 51, ${Math.min(ga, 0.6)})`);
          grd.addColorStop(1, "rgba(255, 51, 51, 0)");
        } else {
          grd.addColorStop(0, `rgba(0, 229, 255, ${Math.min(ga, 0.6)})`);
          grd.addColorStop(1, "rgba(0, 229, 255, 0)");
        }
        ctx!.fillStyle = grd;
        ctx!.fillRect(nx - glR, ny - glR, glR * 2, glR * 2);

        // Core dot
        ctx!.beginPath();
        ctx!.arc(nx, ny, r, 0, Math.PI * 2);
        ctx!.fillStyle = isUser ? USER_COLOR : ACCENT;
        ctx!.fill();

        // Label
        const fs = Math.max(isUser ? 13 : 10, (isUser ? 14 : 11) * scale);
        ctx!.font = isUser
          ? `bold ${fs}px 'JetBrains Mono', monospace`
          : `${fs}px 'JetBrains Mono', monospace`;
        ctx!.fillStyle = isUser
          ? `rgba(255, 51, 51, ${0.7 + pulse * 0.3})`
          : `rgba(0, 229, 255, ${0.5 + pulse * 0.2})`;
        ctx!.textAlign = "center";
        ctx!.fillText(nd.label, nx, ny + r + fs + 2);
      }

      // ── Loading convergence pulse ──────────────────────────────────
      if (state === "loading" && nodes.length > USER_NODE_IDX) {
        const un = nodes[USER_NODE_IDX];
        const cx = projX(un.lon, vp);
        const cy = projY(un.lat, vp);
        const pr = (30 + Math.sin(now * 0.005) * 15) * scale;
        const grd = ctx!.createRadialGradient(cx, cy, 0, cx, cy, pr);
        grd.addColorStop(0, "rgba(255, 51, 51, 0.15)");
        grd.addColorStop(1, "rgba(255, 51, 51, 0)");
        ctx!.fillStyle = grd;
        ctx!.fillRect(cx - pr, cy - pr, pr * 2, pr * 2);
      }

      // ── Edge fades (all 4 sides) ──────────────────────────────────
      // These gradient overlays blend the map into the dark background
      // so there are ZERO hard edges or cutoff lines anywhere.
      const fadeH = vp.mh * 0.08;
      const fadeW = vp.mw * 0.04;
      const bgFull = `rgba(${BG_R}, ${BG_G}, ${BG_B}, 1)`;
      const bgZero = `rgba(${BG_R}, ${BG_G}, ${BG_B}, 0)`;

      // Top
      const tg = ctx!.createLinearGradient(0, vp.oy, 0, vp.oy + fadeH);
      tg.addColorStop(0, bgFull);
      tg.addColorStop(1, bgZero);
      ctx!.fillStyle = tg;
      ctx!.fillRect(0, vp.oy - 2, w, fadeH + 2);

      // Bottom
      const by = vp.oy + vp.mh - fadeH;
      const bg = ctx!.createLinearGradient(0, by, 0, vp.oy + vp.mh);
      bg.addColorStop(0, bgZero);
      bg.addColorStop(1, bgFull);
      ctx!.fillStyle = bg;
      ctx!.fillRect(0, by, w, fadeH + 2);

      // Left
      const lg = ctx!.createLinearGradient(vp.ox, 0, vp.ox + fadeW, 0);
      lg.addColorStop(0, bgFull);
      lg.addColorStop(1, bgZero);
      ctx!.fillStyle = lg;
      ctx!.fillRect(vp.ox - 2, 0, fadeW + 2, h);

      // Right
      const rx = vp.ox + vp.mw - fadeW;
      const rg = ctx!.createLinearGradient(rx, 0, vp.ox + vp.mw, 0);
      rg.addColorStop(0, bgZero);
      rg.addColorStop(1, bgFull);
      ctx!.fillStyle = rg;
      ctx!.fillRect(rx, 0, fadeW + 2, h);

      rafRef.current = requestAnimationFrame(frame);
    }

    rafRef.current = requestAnimationFrame(frame);

    return () => {
      alive = false;
      cancelAnimationFrame(rafRef.current);
      window.removeEventListener("resize", resize);
    };
  }, [resolveState]);

  return (
    <canvas
      ref={canvasRef}
      style={{
        position: "fixed",
        top: 0,
        left: 0,
        width: "100%",
        height: "100%",
        zIndex: 0,
        pointerEvents: "none",
        background: "transparent",
      }}
    />
  );
}
