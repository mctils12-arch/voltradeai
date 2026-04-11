import { useRef, useEffect, useCallback } from "react";
import * as topojson from "topojson-client";
import type { Topology } from "topojson-specification";

// ────────────────────────────────────────────────────────────────────────────
// Types
// ────────────────────────────────────────────────────────────────────────────

interface DataWorldMapProps {
  isLoading: boolean;
  hasData: boolean;
  ticker: string | null;
}

interface Node {
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

// ────────────────────────────────────────────────────────────────────────────
// Data source nodes (lat/lon)
// ────────────────────────────────────────────────────────────────────────────

const DATA_NODES: Node[] = [
  { label: "NYC", lat: 40.71, lon: -74.01, primary: true },
  { label: "CHI", lat: 41.88, lon: -87.63 },
  { label: "SFO", lat: 37.77, lon: -122.42 },
  { label: "LDN", lat: 51.51, lon: -0.13 },
  { label: "FRA", lat: 50.11, lon: 8.68 },
  { label: "TKY", lat: 35.68, lon: 139.69 },
  { label: "HKG", lat: 22.32, lon: 114.17 },
  { label: "SHA", lat: 31.23, lon: 121.47 },
  { label: "MUM", lat: 19.08, lon: 72.88 },
  { label: "SAO", lat: -23.55, lon: -46.63 },
  { label: "SYD", lat: -33.87, lon: 151.21 },
  { label: "TOR", lat: 43.65, lon: -79.38 },
];

// Index 12 will be the user node — added dynamically
const USER_NODE_IDX = 12;

// City-to-city connections (visual complexity)
const CITY_CONNECTIONS: [number, number][] = [
  [4, 3],  // FRA → LDN
  [6, 5],  // HKG → TKY
  [7, 6],  // SHA → HKG
  [8, 3],  // MUM → LDN
  [5, 2],  // TKY → SFO
];

// Connections toward user (built dynamically once user location is known)
function buildUserConnections(): [number, number][] {
  return [
    [0, USER_NODE_IDX],  // NYC → USER
    [1, USER_NODE_IDX],  // CHI → USER
    [2, USER_NODE_IDX],  // SFO → USER
    [3, USER_NODE_IDX],  // LDN → USER
    [4, USER_NODE_IDX],  // FRA → USER
    [5, USER_NODE_IDX],  // TKY → USER
    [6, USER_NODE_IDX],  // HKG → USER
    [8, USER_NODE_IDX],  // MUM → USER
    [9, USER_NODE_IDX],  // SAO → USER
    [10, USER_NODE_IDX], // SYD → USER
    [11, USER_NODE_IDX], // TOR → USER
  ];
}

// ────────────────────────────────────────────────────────────────────────────
// Real geographic data — fetched at runtime from Natural Earth 110m TopoJSON
// ────────────────────────────────────────────────────────────────────────────

const TOPO_URL = "https://cdn.jsdelivr.net/npm/world-atlas@2/land-110m.json";

// Each polygon is an array of rings; ring[0] is the outer boundary, rest are holes
type LandPolygon = number[][][];

// ────────────────────────────────────────────────────────────────────────────
// Projection helpers — full page edge-to-edge
// lon: -180..180 → 0..width
// lat:  ~80..-60 → 0..height (simple equirectangular stretched)
// ────────────────────────────────────────────────────────────────────────────

const LAT_MAX = 85;
const LAT_MIN = -85;

// Aspect-ratio-preserving projection.
// The map's natural ratio is 360 / (LAT_MAX - LAT_MIN).
// We fit it inside the viewport, centered, so continents never stretch.
const MAP_ASPECT = 360 / (LAT_MAX - LAT_MIN); // ~2.57

interface MapViewport {
  offsetX: number;
  offsetY: number;
  mapW: number;
  mapH: number;
}

function getMapViewport(screenW: number, screenH: number): MapViewport {
  const screenAspect = screenW / screenH;
  let mapW: number, mapH: number, offsetX: number, offsetY: number;
  if (screenAspect > MAP_ASPECT) {
    // Wide screen — constrained by height, centered horizontally
    mapH = screenH;
    mapW = mapH * MAP_ASPECT;
    offsetX = (screenW - mapW) / 2;
    offsetY = 0;
  } else {
    // Tall screen (mobile portrait) — fit entire map within width,
    // centered vertically so the whole world is always visible.
    mapW = screenW;
    mapH = mapW / MAP_ASPECT;
    offsetX = 0;
    offsetY = (screenH - mapH) / 2;
  }
  return { offsetX, offsetY, mapW, mapH };
}

function lonToX(lon: number, w: number, vp?: MapViewport): number {
  if (vp) return vp.offsetX + ((lon + 180) / 360) * vp.mapW;
  return ((lon + 180) / 360) * w;
}

function latToY(lat: number, h: number, vp?: MapViewport): number {
  const clamped = Math.max(LAT_MIN, Math.min(LAT_MAX, lat));
  if (vp) return vp.offsetY + ((LAT_MAX - clamped) / (LAT_MAX - LAT_MIN)) * vp.mapH;
  return ((LAT_MAX - clamped) / (LAT_MAX - LAT_MIN)) * h;
}

// ────────────────────────────────────────────────────────────────────────────
// Quadratic bezier point along an arc
// ────────────────────────────────────────────────────────────────────────────

function arcPoint(
  x1: number, y1: number,
  x2: number, y2: number,
  t: number,
  w: number,
): { x: number; y: number } {
  const mx = (x1 + x2) / 2;
  const my = (y1 + y2) / 2;
  const dx = Math.abs(x2 - x1);
  const bulge = Math.min(dx * 0.35, w * 0.12);
  const cx = mx;
  const cy = my - bulge;
  const u = 1 - t;
  return {
    x: u * u * x1 + 2 * u * t * cx + t * t * x2,
    y: u * u * y1 + 2 * u * t * cy + t * t * y2,
  };
}

// ────────────────────────────────────────────────────────────────────────────
// Component
// ────────────────────────────────────────────────────────────────────────────

const ACCENT = "#00e5ff";
const USER_COLOR = "#ff3333";
const BG = "#050a12";
const MAX_PARTICLES = 200;

type AnimState = "idle" | "loading" | "loaded";

export default function DataWorldMap({ isLoading, hasData, ticker }: DataWorldMapProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animRef = useRef<number>(0);
  const stateRef = useRef<AnimState>("idle");
  const prevStateRef = useRef<AnimState>("idle");
  const burstRef = useRef(0);
  const particlesRef = useRef<Particle[]>([]);
  const userLocRef = useRef<{ lat: number; lon: number }>({ lat: 40.71, lon: -74.01 });
  const connectionsRef = useRef<[number, number][]>([]);
  const allNodesRef = useRef<Node[]>([]);
  const landRef = useRef<LandPolygon[]>([]);

  const getState = useCallback((): AnimState => {
    if (isLoading) return "loading";
    if (hasData && ticker) return "loaded";
    return "idle";
  }, [isLoading, hasData, ticker]);

  // Fetch real land polygons once
  useEffect(() => {
    let cancelled = false;
    fetch(TOPO_URL)
      .then((r) => r.json())
      .then((topo: Topology) => {
        if (cancelled) return;
        const land = topojson.feature(topo, topo.objects.land) as GeoJSON.FeatureCollection;
        const polys: LandPolygon[] = [];
        for (const feature of land.features) {
          const geom = feature.geometry;
          if (geom.type === "Polygon") {
            polys.push(geom.coordinates);
          } else if (geom.type === "MultiPolygon") {
            for (const poly of geom.coordinates) {
              polys.push(poly);
            }
          }
        }
        landRef.current = polys;
      })
      .catch(() => {
        // Silently fail — continents just won't render
      });
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let running = true;
    let lastTime = 0;

    // ── Detect user location ──────────────────────────────────────────
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (pos) => {
          userLocRef.current = {
            lat: pos.coords.latitude,
            lon: pos.coords.longitude,
          };
          rebuildNodes();
        },
        () => {
          // Fallback: NYC
          userLocRef.current = { lat: 40.71, lon: -74.01 };
          rebuildNodes();
        },
      );
    }

    function rebuildNodes() {
      const userNode: Node = {
        label: "YOU",
        lat: userLocRef.current.lat,
        lon: userLocRef.current.lon,
        primary: true,
      };
      allNodesRef.current = [...DATA_NODES, userNode];
      connectionsRef.current = [
        ...CITY_CONNECTIONS,
        ...buildUserConnections(),
      ];
    }

    // Initialize with default (NYC)
    rebuildNodes();

    // ── Resize handler ──────────────────────────────────────────────
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

    // ── Particle spawner ────────────────────────────────────────────
    function spawnParticle(): Particle {
      const conns = connectionsRef.current;
      const connIdx = Math.floor(Math.random() * conns.length);
      const [from, to] = conns[connIdx];
      return {
        fromIdx: from,
        toIdx: to,
        t: Math.random(),
        speed: 0.02 + Math.random() * 0.03,
        size: 2 + Math.random() * 2.5,
        opacity: 0.4 + Math.random() * 0.5,
      };
    }

    // Seed initial particles
    if (particlesRef.current.length === 0) {
      for (let i = 0; i < 40; i++) {
        particlesRef.current.push(spawnParticle());
      }
    }

    // ── Animation loop ──────────────────────────────────────────────
    function frame(time: number) {
      if (!running) return;

      const dt = lastTime ? (time - lastTime) / 1000 : 0.016;
      lastTime = time;

      const w = window.innerWidth;
      const h = window.innerHeight;
      const vp = getMapViewport(w, h);
      const nodes = allNodesRef.current;
      const connections = connectionsRef.current;

      const newState = getState();
      if (newState === "loaded" && prevStateRef.current === "loading") {
        burstRef.current = time;
      }
      prevStateRef.current = stateRef.current;
      stateRef.current = newState;

      const state = stateRef.current;
      const burstAge = (time - burstRef.current) / 1000;
      const inBurst = burstAge < 1.5;

      // Speed multiplier & particle targets
      let speedMult = 1;
      let targetParticleCount = 40;
      let nodeGlow = 1;
      let nodePulseSpeed = 1;

      if (state === "loading") {
        speedMult = 4;
        targetParticleCount = 150;
        nodeGlow = 2;
        nodePulseSpeed = 3;
      } else if (state === "loaded") {
        speedMult = inBurst ? 6 : 1.8;
        targetParticleCount = inBurst ? 180 : 60;
        nodeGlow = inBurst ? 2.5 : 1.3;
        nodePulseSpeed = inBurst ? 4 : 1.5;
      }

      // Manage particle count
      const particles = particlesRef.current;
      while (particles.length < targetParticleCount && particles.length < MAX_PARTICLES) {
        particles.push(spawnParticle());
      }
      while (particles.length > targetParticleCount + 10) {
        particles.splice(Math.floor(Math.random() * particles.length), 1);
      }

      // ── Clear ─────────────────────────────────────────────────────
      ctx!.clearRect(0, 0, w, h);

      // ── Draw continent outlines (from real GeoJSON) ──────────────
      const landPolygons = landRef.current;
      if (landPolygons.length > 0) {
        ctx!.fillStyle = "rgba(0, 229, 255, 0.08)";
        ctx!.strokeStyle = "rgba(0, 229, 255, 0.3)";
        ctx!.lineWidth = Math.max(0.8, Math.max(w, h) / 1200);

        // Clip to the map viewport with soft inset to hide boundary artifacts
        const clipInset = vp.mapH * 0.02; // 2% fade zone
        ctx!.save();
        ctx!.beginPath();
        ctx!.rect(vp.offsetX, vp.offsetY + clipInset, vp.mapW, vp.mapH - clipInset * 2);
        ctx!.clip();

        for (const polygon of landPolygons) {
          const outerRing = polygon[0];
          if (!outerRing || outerRing.length < 2) continue;

          // Skip Antarctica polygons (they create horizontal border lines)
          const isAntarctica = outerRing.some((c: number[]) => c[1] < -70);
          if (isAntarctica) continue;

          ctx!.beginPath();
          ctx!.moveTo(lonToX(outerRing[0][0], w, vp), latToY(outerRing[0][1], h, vp));
          for (let i = 1; i < outerRing.length; i++) {
            ctx!.lineTo(lonToX(outerRing[i][0], w, vp), latToY(outerRing[i][1], h, vp));
          }
          ctx!.closePath();
          ctx!.fill();
          ctx!.stroke();
        }

        ctx!.restore();
      }

      // ── Edge fade: top and bottom gradient masks ──────────────────
      const fadeH = vp.mapH * 0.06;
      // Top fade
      const topGrad = ctx!.createLinearGradient(0, vp.offsetY, 0, vp.offsetY + fadeH);
      topGrad.addColorStop(0, "rgba(5, 10, 18, 1)");
      topGrad.addColorStop(1, "rgba(5, 10, 18, 0)");
      ctx!.fillStyle = topGrad;
      ctx!.fillRect(0, vp.offsetY, w, fadeH);
      // Bottom fade
      const btmY = vp.offsetY + vp.mapH - fadeH;
      const btmGrad = ctx!.createLinearGradient(0, btmY, 0, btmY + fadeH);
      btmGrad.addColorStop(0, "rgba(5, 10, 18, 0)");
      btmGrad.addColorStop(1, "rgba(5, 10, 18, 1)");
      ctx!.fillStyle = btmGrad;
      ctx!.fillRect(0, btmY, w, fadeH);

      // ── Draw connection base lines ────────────────────────────────
      for (const [fromIdx, toIdx] of connections) {
        if (fromIdx >= nodes.length || toIdx >= nodes.length) continue;
        const from = nodes[fromIdx];
        const to = nodes[toIdx];
        const x1 = lonToX(from.lon, w, vp);
        const y1 = latToY(from.lat, h, vp);
        const x2 = lonToX(to.lon, w, vp);
        const y2 = latToY(to.lat, h, vp);

        const lineAlpha = state === "loading" ? 0.2 : state === "loaded" && inBurst ? 0.2 : 0.1;

        ctx!.beginPath();
        const mx = (x1 + x2) / 2;
        const my = (y1 + y2) / 2;
        const dx = Math.abs(x2 - x1);
        const bulge = Math.min(dx * 0.35, vp.mapW * 0.12);
        ctx!.moveTo(x1, y1);
        ctx!.quadraticCurveTo(mx, my - bulge, x2, y2);
        ctx!.strokeStyle = `rgba(0, 229, 255, ${lineAlpha})`;
        ctx!.lineWidth = 1;
        ctx!.stroke();
      }

      // ── Draw particles ────────────────────────────────────────────
      for (let i = particles.length - 1; i >= 0; i--) {
        const p = particles[i];
        p.t += p.speed * speedMult * dt;

        if (p.t >= 1) {
          particles[i] = spawnParticle();
          particles[i].t = 0;
          continue;
        }

        if (p.fromIdx >= nodes.length || p.toIdx >= nodes.length) continue;
        const from = nodes[p.fromIdx];
        const to = nodes[p.toIdx];
        const x1 = lonToX(from.lon, w, vp);
        const y1 = latToY(from.lat, h, vp);
        const x2 = lonToX(to.lon, w, vp);
        const y2 = latToY(to.lat, h, vp);

        const pos = arcPoint(x1, y1, x2, y2, p.t, vp.mapW);

        // Glow
        const glowRadius = (p.size + 3) * (state === "loading" ? 1.5 : 1);
        const gradient = ctx!.createRadialGradient(pos.x, pos.y, 0, pos.x, pos.y, glowRadius);
        gradient.addColorStop(0, `rgba(0, 229, 255, ${p.opacity * 0.6})`);
        gradient.addColorStop(1, "rgba(0, 229, 255, 0)");
        ctx!.fillStyle = gradient;
        ctx!.fillRect(pos.x - glowRadius, pos.y - glowRadius, glowRadius * 2, glowRadius * 2);

        // Core dot
        ctx!.beginPath();
        ctx!.arc(pos.x, pos.y, p.size * 0.6, 0, Math.PI * 2);
        ctx!.fillStyle = `rgba(0, 229, 255, ${p.opacity})`;
        ctx!.fill();
      }

      // ── Draw burst ripple ─────────────────────────────────────────
      if (inBurst && nodes.length > USER_NODE_IDX) {
        const userNode = nodes[USER_NODE_IDX];
        const centerX = lonToX(userNode.lon, w, vp);
        const centerY = latToY(userNode.lat, h, vp);
        const rippleRadius = burstAge * 400;
        const rippleAlpha = Math.max(0, 0.3 * (1 - burstAge / 1.5));
        ctx!.beginPath();
        ctx!.arc(centerX, centerY, rippleRadius, 0, Math.PI * 2);
        ctx!.strokeStyle = `rgba(255, 51, 51, ${rippleAlpha})`;
        ctx!.lineWidth = 2;
        ctx!.stroke();
      }

      // ── Draw data source nodes ────────────────────────────────────
      const pulseT = Math.sin(time * 0.001 * nodePulseSpeed) * 0.5 + 0.5;

      for (let n = 0; n < nodes.length; n++) {
        const node = nodes[n];
        const x = lonToX(node.lon, w, vp);
        const y = latToY(node.lat, h, vp);
        const isUser = n === USER_NODE_IDX;

        // Scale node sizes relative to viewport so they're visible on all screens
        const scale = Math.max(w, h) / 1000;
        const baseSize = (isUser ? 8 : node.primary ? 6 : 4) * scale;
        const pulseSize = baseSize + pulseT * 3 * nodeGlow * scale;

        // Outer glow
        const glowR = pulseSize * (isUser ? 5 : 3.5);
        const grd = ctx!.createRadialGradient(x, y, 0, x, y, glowR);
        const glowAlpha = (0.15 + pulseT * 0.15) * nodeGlow;
        if (isUser) {
          grd.addColorStop(0, `rgba(255, 51, 51, ${Math.min(glowAlpha, 0.6)})`);
          grd.addColorStop(1, "rgba(255, 51, 51, 0)");
        } else {
          grd.addColorStop(0, `rgba(0, 229, 255, ${Math.min(glowAlpha, 0.6)})`);
          grd.addColorStop(1, "rgba(0, 229, 255, 0)");
        }
        ctx!.fillStyle = grd;
        ctx!.fillRect(x - glowR, y - glowR, glowR * 2, glowR * 2);

        // Core
        ctx!.beginPath();
        ctx!.arc(x, y, pulseSize, 0, Math.PI * 2);
        ctx!.fillStyle = isUser ? USER_COLOR : ACCENT;
        ctx!.fill();

        // Label
        const fontSize = Math.max(isUser ? 13 : 10, (isUser ? 14 : 11) * scale);
        ctx!.font = isUser
          ? `bold ${fontSize}px 'JetBrains Mono', monospace`
          : `${fontSize}px 'JetBrains Mono', monospace`;
        ctx!.fillStyle = isUser
          ? `rgba(255, 51, 51, ${0.7 + pulseT * 0.3})`
          : `rgba(0, 229, 255, ${0.5 + pulseT * 0.2})`;
        ctx!.textAlign = "center";
        ctx!.fillText(node.label, x, y + pulseSize + fontSize + 2);
      }

      // ── Loading center convergence effect ─────────────────────────
      if (state === "loading" && nodes.length > USER_NODE_IDX) {
        const userNode = nodes[USER_NODE_IDX];
        const cx = lonToX(userNode.lon, w, vp);
        const cy = latToY(userNode.lat, h, vp);
        const cScale = Math.max(w, h) / 1000;
        const pulseRadius = (30 + Math.sin(time * 0.005) * 15) * cScale;
        const grd = ctx!.createRadialGradient(cx, cy, 0, cx, cy, pulseRadius);
        grd.addColorStop(0, "rgba(255, 51, 51, 0.15)");
        grd.addColorStop(1, "rgba(255, 51, 51, 0)");
        ctx!.fillStyle = grd;
        ctx!.fillRect(cx - pulseRadius, cy - pulseRadius, pulseRadius * 2, pulseRadius * 2);
      }

      animRef.current = requestAnimationFrame(frame);
    }

    animRef.current = requestAnimationFrame(frame);

    return () => {
      running = false;
      cancelAnimationFrame(animRef.current);
      window.removeEventListener("resize", resize);
    };
  }, [getState]);

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
        background: BG,
      }}
    />
  );
}
