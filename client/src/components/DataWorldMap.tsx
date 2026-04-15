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

const LAT_MAX = 85;
const LAT_MIN = -85;
const MAP_ASPECT = 360 / (LAT_MAX - LAT_MIN); // ~2.12 (full globe)

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
  const bulge = Math.min(Math.abs(x2 - x1) * 0.45, mapW * 0.18);
  const cx = mx;
  const cy = my - bulge;
  const u = 1 - t;
  return {
    x: u * u * x1 + 2 * u * t * cx + t * t * x2,
    y: u * u * y1 + 2 * u * t * cy + t * t * y2,
  };
}

// ── Offscreen sprite helpers ────────────────────────────────────────────────

/** Pre-render a soft radial glow dot to an offscreen canvas for reuse */
function createGlowSprite(
  radius: number,
  r: number, g: number, b: number,
  peakAlpha: number,
): HTMLCanvasElement {
  const size = Math.ceil(radius * 2);
  const c = document.createElement("canvas");
  c.width = size;
  c.height = size;
  const cx = c.getContext("2d")!;
  const grad = cx.createRadialGradient(radius, radius, 0, radius, radius, radius);
  grad.addColorStop(0, `rgba(${r}, ${g}, ${b}, ${peakAlpha})`);
  grad.addColorStop(1, `rgba(${r}, ${g}, ${b}, 0)`);
  cx.fillStyle = grad;
  cx.fillRect(0, 0, size, size);
  return c;
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
    const ctx = canvas.getContext("2d", { alpha: true });
    if (!ctx) return;

    let alive = true;
    let lastT = 0;

    // ── Cached offscreen layers ──────────────────────────────────────
    let landCanvas: HTMLCanvasElement | null = null;
    let edgeFadeCanvas: HTMLCanvasElement | null = null;
    let cachedW = 0;
    let cachedH = 0;
    let cachedVp: Viewport | null = null;

    // Pre-rendered glow sprites (keyed by "r,g,b,size")
    const glowSpriteCache = new Map<string, HTMLCanvasElement>();

    function getCachedGlowSprite(
      radius: number, r: number, g: number, b: number, peakAlpha: number,
    ): HTMLCanvasElement {
      // Bucket radius to reduce sprite count (round to nearest integer)
      const bucketR = Math.max(1, Math.round(radius));
      const key = `${r},${g},${b},${bucketR}`;
      let sprite = glowSpriteCache.get(key);
      if (!sprite) {
        sprite = createGlowSprite(bucketR, r, g, b, peakAlpha);
        glowSpriteCache.set(key, sprite);
      }
      return sprite;
    }

    // ── Render land mass + coastlines to offscreen canvas ────────────
    function renderLandLayer(w: number, h: number, vp: Viewport) {
      const polys = landRef.current;
      if (polys.length === 0) return;

      const dpr = window.devicePixelRatio || 1;
      if (!landCanvas) {
        landCanvas = document.createElement("canvas");
      }
      landCanvas.width = w * dpr;
      landCanvas.height = h * dpr;
      const lctx = landCanvas.getContext("2d")!;
      lctx.setTransform(dpr, 0, 0, dpr, 0, 0);

      // Continent fills
      lctx.fillStyle = "rgba(0, 229, 255, 0.08)";
      for (const poly of polys) {
        const ring = poly[0];
        if (!ring || ring.length < 3) continue;
        lctx.beginPath();
        let penDown = false;
        for (let i = 0; i < ring.length; i++) {
          const j = (i + 1) % ring.length;
          const px = projX(ring[i][0], vp);
          const py = projYRaw(ring[i][1], vp);
          const lonSpan = i < ring.length - 1 ? Math.abs(ring[j][0] - ring[i][0]) : 0;

          if (!penDown) { lctx.moveTo(px, py); penDown = true; }
          else lctx.lineTo(px, py);

          if (lonSpan > 180) {
            lctx.closePath();
            lctx.fill();
            lctx.beginPath();
            penDown = false;
          }
        }
        lctx.closePath();
        lctx.fill();
      }

      // Coastline strokes
      lctx.strokeStyle = "rgba(0, 229, 255, 0.25)";
      lctx.lineWidth = Math.max(0.8, Math.max(w, h) / 1200);

      for (const poly of polys) {
        const ring = poly[0];
        if (!ring || ring.length < 3) continue;
        lctx.beginPath();
        let penDown = false;

        for (let i = 0; i < ring.length; i++) {
          const j = (i + 1) % ring.length;
          const lat1 = ring[i][1];
          const lat2 = ring[j][1];

          if (lat1 > 83 && lat2 > 83 && Math.abs(lat1 - lat2) < 1) {
            penDown = false;
            continue;
          }

          const lonSpan = Math.abs(ring[j][0] - ring[i][0]);
          if (lonSpan > 180) {
            penDown = false;
            continue;
          }

          const x1 = projX(ring[i][0], vp);
          const y1 = projYRaw(lat1, vp);

          if (!penDown) { lctx.moveTo(x1, y1); penDown = true; }
          else lctx.lineTo(x1, y1);
        }
        lctx.stroke();
      }
    }

    // ── Render edge fades to offscreen canvas ───────────────────────
    function renderEdgeFades(w: number, h: number, vp: Viewport) {
      const dpr = window.devicePixelRatio || 1;
      if (!edgeFadeCanvas) {
        edgeFadeCanvas = document.createElement("canvas");
      }
      edgeFadeCanvas.width = w * dpr;
      edgeFadeCanvas.height = h * dpr;
      const ectx = edgeFadeCanvas.getContext("2d")!;
      ectx.setTransform(dpr, 0, 0, dpr, 0, 0);

      const fadeH = vp.mh * 0.06;
      const fadeW = vp.mw * 0.03;
      const bgFull = `rgba(${BG_R}, ${BG_G}, ${BG_B}, 1)`;
      const bgZero = `rgba(${BG_R}, ${BG_G}, ${BG_B}, 0)`;

      // Top
      const tg = ectx.createLinearGradient(0, vp.oy, 0, vp.oy + fadeH);
      tg.addColorStop(0, bgFull);
      tg.addColorStop(1, bgZero);
      ectx.fillStyle = tg;
      ectx.fillRect(0, vp.oy - 2, w, fadeH + 2);

      // Bottom
      const by = vp.oy + vp.mh - fadeH;
      const bg = ectx.createLinearGradient(0, by, 0, vp.oy + vp.mh);
      bg.addColorStop(0, bgZero);
      bg.addColorStop(1, bgFull);
      ectx.fillStyle = bg;
      ectx.fillRect(0, by, w, fadeH + 2);

      // Left
      const lg = ectx.createLinearGradient(vp.ox, 0, vp.ox + fadeW, 0);
      lg.addColorStop(0, bgFull);
      lg.addColorStop(1, bgZero);
      ectx.fillStyle = lg;
      ectx.fillRect(vp.ox - 2, 0, fadeW + 2, h);

      // Right
      const rx = vp.ox + vp.mw - fadeW;
      const rg = ectx.createLinearGradient(rx, 0, vp.ox + vp.mw, 0);
      rg.addColorStop(0, bgZero);
      rg.addColorStop(1, bgFull);
      ectx.fillStyle = rg;
      ectx.fillRect(rx, 0, fadeW + 2, h);
    }

    // -- Build node list & connections --
    function rebuild() {
      const user: CityNode = {
        label: "YOU",
        lat: userLocRef.current.lat,
        lon: userLocRef.current.lon,
        primary: true,
      };
      nodesRef.current = [...DATA_NODES, user];
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
      // Invalidate caches on resize
      cachedW = 0;
      cachedH = 0;
      glowSpriteCache.clear();
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

      if (state === "loading") {
        speedMul = 4;
        targetCount = 150;
        glowMul = 2;
      } else if (state === "loaded") {
        speedMul = inBurst ? 6 : 1.8;
        targetCount = inBurst ? 180 : 60;
        glowMul = inBurst ? 2.5 : 1.3;
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

      // ── Draw cached land layer (re-render only on resize or first load) ──
      if (landRef.current.length > 0 && (w !== cachedW || h !== cachedH)) {
        renderLandLayer(w, h, vp);
        renderEdgeFades(w, h, vp);
        cachedW = w;
        cachedH = h;
        cachedVp = vp;
      }
      if (landCanvas) {
        ctx!.drawImage(landCanvas, 0, 0, w, h);
      }

      // ── Data flow arc lines (curved paths between cities) ────────
      if (nodes.length > 0) {
        const conns = connsRef.current;
        const lineAlpha = state === "loading" ? 0.35
          : (state === "loaded" && inBurst) ? 0.35
          : 0.25;
        ctx!.strokeStyle = `rgba(0, 229, 255, ${lineAlpha})`;
        ctx!.lineWidth = 1.2;
        // Batch all arcs into a single path for fewer draw calls
        ctx!.beginPath();
        for (const [fi, ti] of conns) {
          if (fi >= nodes.length || ti >= nodes.length) continue;
          const from = nodes[fi];
          const to = nodes[ti];
          const x1 = projX(from.lon, vp);
          const y1 = projY(from.lat, vp);
          const x2 = projX(to.lon, vp);
          const y2 = projY(to.lat, vp);
          const mx = (x1 + x2) / 2;
          const my = (y1 + y2) / 2;
          const dx = Math.abs(x2 - x1);
          const bulge = Math.min(dx * 0.45, vp.mw * 0.18);
          ctx!.moveTo(x1, y1);
          ctx!.quadraticCurveTo(mx, my - bulge, x2, y2);
        }
        ctx!.stroke();
      }

      // ── Particles (flowing dots along arc paths) ───────────────────
      // Draw all particle glow sprites first, then dots on top
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

        // Glow via pre-rendered sprite (avoid per-frame createRadialGradient)
        const glowR = (p.size + 3) * (state === "loading" ? 1.5 : 1);
        const sprite = getCachedGlowSprite(glowR, 0, 229, 255, 0.6);
        const spriteR = sprite.width / 2;
        ctx!.globalAlpha = p.opacity;
        ctx!.drawImage(sprite, pt.x - spriteR, pt.y - spriteR, spriteR * 2, spriteR * 2);

        // Dot
        ctx!.beginPath();
        ctx!.arc(pt.x, pt.y, p.size * 0.6, 0, Math.PI * 2);
        ctx!.fillStyle = ACCENT;
        ctx!.fill();
      }
      ctx!.globalAlpha = 1;

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
      const scale = Math.max(w, h) / 1000;

      // Collect label rects for collision detection (Fix 2 & 3)
      type LabelRect = {
        idx: number;
        nx: number;
        ny: number;
        r: number;
        fs: number;
        label: string;
        isUser: boolean;
        x: number;
        y: number;
        w: number;
        h: number;
      };
      const labelRects: LabelRect[] = [];

      for (let n = 0; n < nodes.length; n++) {
        const nd = nodes[n];
        const nx = projX(nd.lon, vp);
        const ny = projY(nd.lat, vp);
        const isUser = n === USER_NODE_IDX;

        // Fixed dot radius — uniform for all nodes
        const r = 3 * scale;

        // Outer glow — fixed alpha, no pulse
        const glR = r * 3.5;
        const ga = 0.22 * glowMul;
        const nodeSprite = isUser
          ? getCachedGlowSprite(glR, 255, 51, 51, 1)
          : getCachedGlowSprite(glR, 0, 229, 255, 1);
        ctx!.globalAlpha = Math.min(ga, 0.6);
        ctx!.drawImage(nodeSprite, nx - glR, ny - glR, glR * 2, glR * 2);
        ctx!.globalAlpha = 1;

        // Core dot
        ctx!.beginPath();
        ctx!.arc(nx, ny, r, 0, Math.PI * 2);
        ctx!.fillStyle = isUser ? USER_COLOR : ACCENT;
        ctx!.fill();

        // Compute label metrics for collision detection
        const fs = Math.max(isUser ? 13 : 10, (isUser ? 14 : 11) * scale);
        ctx!.font = isUser
          ? `bold ${fs}px 'JetBrains Mono', monospace`
          : `${fs}px 'JetBrains Mono', monospace`;
        const tw = ctx!.measureText(nd.label).width;
        // Default position: below the dot, centered
        const lx = nx - tw / 2;
        const ly = ny + r + fs + 2;
        labelRects.push({
          idx: n, nx, ny, r, fs, label: nd.label, isUser,
          x: lx, y: ly - fs, w: tw, h: fs + 4,
        });
      }

      // ── Label collision detection & resolution ─────────────────────
      // "YOU" label always gets priority; nudge others on overlap
      // Sort so USER is processed first (its position is locked)
      labelRects.sort((a, b) => (a.isUser ? -1 : b.isUser ? 1 : 0));

      function rectsOverlap(a: LabelRect, b: LabelRect): boolean {
        return a.x < b.x + b.w && a.x + a.w > b.x &&
               a.y < b.y + b.h && a.y + a.h > b.y;
      }

      // Alternate positions: above, left, right
      function tryAlternatePositions(rect: LabelRect, settled: LabelRect[]): void {
        const positions = [
          // Above the dot
          { x: rect.nx - rect.w / 2, y: rect.ny - rect.r - rect.fs - 2 },
          // Right of the dot
          { x: rect.nx + rect.r + 2, y: rect.ny + rect.fs / 3 },
          // Left of the dot
          { x: rect.nx - rect.r - rect.w - 2, y: rect.ny + rect.fs / 3 },
          // Further below
          { x: rect.nx - rect.w / 2, y: rect.ny + rect.r + rect.fs + 4 },
        ];

        for (const pos of positions) {
          const candidate = { ...rect, x: pos.x, y: pos.y - rect.fs };
          let overlap = false;
          for (const s of settled) {
            if (rectsOverlap(candidate, s)) { overlap = true; break; }
          }
          if (!overlap) {
            rect.x = candidate.x;
            rect.y = candidate.y;
            return;
          }
        }
        // If all positions overlap, keep the last tried (further below)
        const last = positions[positions.length - 1];
        rect.x = last.x;
        rect.y = last.y - rect.fs;
      }

      const settled: LabelRect[] = [];
      for (const rect of labelRects) {
        let hasOverlap = false;
        for (const s of settled) {
          if (rectsOverlap(rect, s)) { hasOverlap = true; break; }
        }
        if (hasOverlap && !rect.isUser) {
          tryAlternatePositions(rect, settled);
        }
        settled.push(rect);
      }

      // Draw all labels with fixed opacity (no pulse)
      for (const lr of labelRects) {
        ctx!.font = lr.isUser
          ? `bold ${lr.fs}px 'JetBrains Mono', monospace`
          : `${lr.fs}px 'JetBrains Mono', monospace`;
        ctx!.fillStyle = lr.isUser
          ? `rgba(255, 51, 51, 0.85)`
          : `rgba(0, 229, 255, 0.6)`;
        ctx!.textAlign = "left";
        ctx!.fillText(lr.label, lr.x, lr.y + lr.fs);
      }

      // ── Loading convergence pulse ──────────────────────────────────
      if (state === "loading" && nodes.length > USER_NODE_IDX) {
        const un = nodes[USER_NODE_IDX];
        const cx = projX(un.lon, vp);
        const cy = projY(un.lat, vp);
        const pr = (30 + Math.sin(now * 0.005) * 15) * scale;
        const pulseSprite = getCachedGlowSprite(pr, 255, 51, 51, 0.15);
        ctx!.drawImage(pulseSprite, cx - pr, cy - pr, pr * 2, pr * 2);
      }

      // ── Edge fades (cached offscreen) ──────────────────────────────
      if (edgeFadeCanvas) {
        ctx!.drawImage(edgeFadeCanvas, 0, 0, w, h);
      }

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
