import { useRef, useEffect, useCallback } from "react";

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
  t: number; // 0..1 progress along arc
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

// Connections: pairs of node indices forming arcs (all converge toward NYC)
const CONNECTIONS: [number, number][] = [
  [1, 0],  // CHI → NYC
  [2, 0],  // SFO → NYC
  [3, 0],  // LDN → NYC
  [4, 3],  // FRA → LDN
  [5, 0],  // TKY → NYC
  [6, 5],  // HKG → TKY
  [7, 6],  // SHA → HKG
  [8, 3],  // MUM → LDN
  [9, 0],  // SAO → NYC
  [10, 6], // SYD → HKG
  [11, 0], // TOR → NYC
  [4, 0],  // FRA → NYC
  [8, 0],  // MUM → NYC
  [5, 2],  // TKY → SFO
  [10, 2], // SYD → SFO
];

// ────────────────────────────────────────────────────────────────────────────
// Simplified world map dot coordinates (lon, lat pairs for continents)
// Mercator-projected dot-matrix outline of continents
// ────────────────────────────────────────────────────────────────────────────

const WORLD_DOTS: [number, number][] = generateWorldDots();

function generateWorldDots(): [number, number][] {
  const dots: [number, number][] = [];

  // Each continent defined as polygon-ish outlines, dots placed along them
  const continents: [number, number][][] = [
    // North America outline
    [
      [-130, 50], [-125, 55], [-120, 60], [-115, 62], [-110, 60],
      [-100, 60], [-95, 55], [-85, 52], [-80, 45], [-75, 40],
      [-80, 35], [-85, 30], [-90, 30], [-95, 28], [-100, 25],
      [-105, 20], [-100, 18], [-95, 18], [-90, 20], [-85, 15],
      [-80, 10], [-120, 35], [-115, 32], [-110, 30], [-105, 30],
      [-100, 30], [-95, 35], [-90, 35], [-85, 35], [-80, 38],
      [-75, 42], [-70, 44], [-65, 47], [-60, 50], [-55, 52],
      // Canada fill
      [-130, 55], [-125, 58], [-120, 55], [-115, 55], [-110, 55],
      [-105, 55], [-100, 55], [-95, 50], [-90, 48], [-85, 48],
      [-80, 50], [-75, 50], [-70, 50], [-65, 50],
      // US interior
      [-120, 42], [-115, 42], [-110, 42], [-105, 42], [-100, 42],
      [-95, 42], [-90, 42], [-85, 42], [-120, 38], [-115, 38],
      [-110, 38], [-105, 38], [-100, 38], [-95, 38], [-90, 38],
      [-105, 35], [-100, 35], [-95, 32], [-90, 32],
    ],
    // South America outline
    [
      [-80, 8], [-75, 5], [-70, 5], [-65, 0], [-60, -3],
      [-55, -5], [-50, -3], [-45, -5], [-40, -8], [-38, -12],
      [-40, -15], [-42, -20], [-45, -22], [-48, -25],
      [-50, -28], [-53, -30], [-55, -33], [-57, -35],
      [-60, -38], [-65, -40], [-68, -45], [-70, -48],
      [-73, -45], [-72, -40], [-70, -35], [-70, -30],
      [-70, -25], [-70, -20], [-72, -15], [-75, -10],
      [-78, -5], [-77, 0], [-75, 2],
      // Interior fill
      [-60, -10], [-55, -12], [-50, -15], [-48, -18],
      [-55, -20], [-60, -25], [-65, -30], [-60, -32],
    ],
    // Europe outline
    [
      [-10, 38], [-5, 38], [0, 40], [2, 43], [5, 44],
      [10, 44], [12, 42], [15, 40], [18, 40], [20, 38],
      [25, 38], [28, 40], [30, 42], [28, 44], [25, 46],
      [20, 48], [15, 48], [12, 50], [10, 52], [8, 54],
      [10, 55], [12, 56], [15, 58], [18, 60], [20, 62],
      [25, 62], [28, 60], [30, 60], [32, 58], [30, 55],
      [25, 55], [20, 55], [15, 54], [12, 54], [10, 52],
      [5, 48], [2, 48], [0, 50], [-5, 52], [-8, 54],
      [-5, 56], [-3, 58], [0, 58], [-8, 50], [-10, 42],
      // Interior
      [5, 50], [8, 50], [12, 48], [15, 46], [18, 44],
      [20, 42], [22, 44], [25, 42], [20, 50], [22, 52],
      [25, 50], [28, 48], [30, 50],
    ],
    // Africa outline
    [
      [-15, 12], [-10, 15], [-5, 15], [0, 8], [5, 5],
      [10, 5], [12, 2], [15, 2], [20, 0], [25, -2],
      [30, -5], [33, -10], [35, -15], [38, -18],
      [35, -22], [32, -25], [30, -28], [28, -32],
      [25, -34], [22, -34], [18, -30], [15, -25],
      [12, -15], [10, -5], [8, 0], [5, 2],
      [0, 5], [-5, 8], [-10, 10], [-15, 10],
      [-18, 15], [-15, 18], [-10, 20], [-5, 22],
      [0, 25], [5, 28], [10, 30], [15, 32],
      [20, 32], [25, 30], [30, 30], [33, 28],
      // Interior fill
      [15, 10], [20, 8], [25, 5], [20, 15], [25, 15],
      [30, 10], [15, -5], [20, -10], [25, -15],
      [30, -20], [20, -20], [15, -15], [10, -10],
    ],
    // Asia outline
    [
      [35, 35], [40, 38], [45, 40], [50, 40], [55, 42],
      [60, 42], [65, 45], [70, 45], [75, 42], [80, 38],
      [85, 28], [88, 22], [92, 20], [98, 18], [100, 15],
      [105, 12], [108, 10], [110, 15], [115, 20], [120, 25],
      [122, 28], [125, 32], [128, 35], [130, 38], [132, 40],
      [135, 38], [140, 40], [142, 42], [145, 44],
      // Russia / north
      [40, 55], [45, 55], [50, 55], [55, 55], [60, 55],
      [65, 55], [70, 55], [75, 58], [80, 58], [85, 55],
      [90, 55], [95, 52], [100, 52], [105, 55], [110, 55],
      [115, 52], [120, 50], [125, 48], [130, 48], [135, 50],
      [140, 48], [145, 50], [150, 55], [155, 55], [160, 58],
      [165, 62], [170, 62],
      // Central
      [50, 45], [55, 48], [60, 48], [65, 50], [70, 50],
      [75, 50], [80, 48], [85, 45], [90, 45], [95, 45],
      [100, 45], [105, 45], [110, 45], [115, 42], [120, 38],
      [125, 38], [120, 32], [115, 28], [110, 22], [105, 18],
      [100, 22], [95, 25], [90, 28], [85, 32], [80, 32],
      [75, 35], [70, 35], [65, 35], [60, 35],
    ],
    // Australia outline
    [
      [115, -14], [120, -14], [125, -14], [130, -12],
      [135, -12], [140, -15], [145, -15], [148, -18],
      [150, -22], [152, -25], [153, -28], [150, -32],
      [148, -35], [145, -38], [140, -38], [138, -35],
      [135, -35], [132, -32], [128, -32], [125, -30],
      [120, -28], [115, -25], [115, -20],
      // Interior
      [125, -20], [130, -20], [135, -22], [140, -22],
      [145, -25], [140, -28], [135, -28], [130, -25],
      [125, -25], [120, -22],
    ],
  ];

  for (const continent of continents) {
    for (const [lon, lat] of continent) {
      dots.push([lon, lat]);
    }
  }

  return dots;
}

// ────────────────────────────────────────────────────────────────────────────
// Mercator projection helpers
// ────────────────────────────────────────────────────────────────────────────

function lonToX(lon: number, w: number): number {
  return ((lon + 180) / 360) * w;
}

function latToY(lat: number, h: number): number {
  // Simple Mercator: clamp between -80 and 80
  const latRad = (Math.max(-80, Math.min(80, lat)) * Math.PI) / 180;
  const mercN = Math.log(Math.tan(Math.PI / 4 + latRad / 2));
  // Normalize to 0..1 (where -80 → 1, 80 → 0)
  const yNorm = 0.5 - mercN / (2 * Math.PI);
  return yNorm * h;
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
  // Control point: midpoint raised upward proportional to distance
  const mx = (x1 + x2) / 2;
  const my = (y1 + y2) / 2;
  const dx = Math.abs(x2 - x1);
  // Handle wrap-around for connections that cross the date line
  const bulge = Math.min(dx * 0.35, w * 0.12);
  const cx = mx;
  const cy = my - bulge;
  // Quadratic bezier
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
const GREEN = "#00ff41";
const BG = "#050a12";
const MAX_PARTICLES = 200;

type AnimState = "idle" | "loading" | "loaded";

export default function DataWorldMap({ isLoading, hasData, ticker }: DataWorldMapProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animRef = useRef<number>(0);
  const stateRef = useRef<AnimState>("idle");
  const prevStateRef = useRef<AnimState>("idle");
  const burstRef = useRef(0); // timestamp of last burst
  const particlesRef = useRef<Particle[]>([]);

  // Determine animation state
  const getState = useCallback((): AnimState => {
    if (isLoading) return "loading";
    if (hasData && ticker) return "loaded";
    return "idle";
  }, [isLoading, hasData, ticker]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let running = true;
    let lastTime = 0;

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

    // ── Initialize particles ────────────────────────────────────────
    function spawnParticle(): Particle {
      const connIdx = Math.floor(Math.random() * CONNECTIONS.length);
      const [from, to] = CONNECTIONS[connIdx];
      return {
        fromIdx: from,
        toIdx: to,
        t: Math.random(),
        speed: 0.02 + Math.random() * 0.03,
        size: 1 + Math.random() * 1.5,
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

      const newState = getState();
      // Detect transition to "loaded" → trigger burst
      if (newState === "loaded" && prevStateRef.current === "loading") {
        burstRef.current = time;
      }
      prevStateRef.current = stateRef.current;
      stateRef.current = newState;

      const state = stateRef.current;
      const burstAge = (time - burstRef.current) / 1000; // seconds since burst
      const inBurst = burstAge < 1.5;

      // Speed multiplier
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
      // Remove excess particles gradually
      while (particles.length > targetParticleCount + 10) {
        particles.splice(Math.floor(Math.random() * particles.length), 1);
      }

      // ── Clear ───────────────────────────────────────────────────
      ctx!.clearRect(0, 0, w, h);

      // ── Draw world dots ─────────────────────────────────────────
      ctx!.fillStyle = "rgba(0, 229, 255, 0.12)";
      for (const [lon, lat] of WORLD_DOTS) {
        const x = lonToX(lon, w);
        const y = latToY(lat, h);
        ctx!.fillRect(x, y, 1.5, 1.5);
      }

      // ── Draw connection base lines ──────────────────────────────
      for (const [fromIdx, toIdx] of CONNECTIONS) {
        const from = DATA_NODES[fromIdx];
        const to = DATA_NODES[toIdx];
        const x1 = lonToX(from.lon, w);
        const y1 = latToY(from.lat, h);
        const x2 = lonToX(to.lon, w);
        const y2 = latToY(to.lat, h);

        const lineAlpha = state === "loading" ? 0.08 : state === "loaded" && inBurst ? 0.1 : 0.04;

        ctx!.beginPath();
        const mx = (x1 + x2) / 2;
        const my = (y1 + y2) / 2;
        const dx = Math.abs(x2 - x1);
        const bulge = Math.min(dx * 0.35, w * 0.12);
        ctx!.moveTo(x1, y1);
        ctx!.quadraticCurveTo(mx, my - bulge, x2, y2);
        ctx!.strokeStyle = `rgba(0, 229, 255, ${lineAlpha})`;
        ctx!.lineWidth = 1;
        ctx!.stroke();
      }

      // ── Draw particles ──────────────────────────────────────────
      for (let i = particles.length - 1; i >= 0; i--) {
        const p = particles[i];
        p.t += p.speed * speedMult * dt;

        if (p.t >= 1) {
          // Respawn
          particles[i] = spawnParticle();
          particles[i].t = 0;
          continue;
        }

        const from = DATA_NODES[p.fromIdx];
        const to = DATA_NODES[p.toIdx];
        const x1 = lonToX(from.lon, w);
        const y1 = latToY(from.lat, h);
        const x2 = lonToX(to.lon, w);
        const y2 = latToY(to.lat, h);

        const pos = arcPoint(x1, y1, x2, y2, p.t, w);

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

      // ── Draw burst ripple ───────────────────────────────────────
      if (inBurst) {
        const centerX = lonToX(-40, w); // Roughly Atlantic center
        const centerY = latToY(30, h);
        const rippleRadius = burstAge * 400;
        const rippleAlpha = Math.max(0, 0.3 * (1 - burstAge / 1.5));
        ctx!.beginPath();
        ctx!.arc(centerX, centerY, rippleRadius, 0, Math.PI * 2);
        ctx!.strokeStyle = `rgba(0, 229, 255, ${rippleAlpha})`;
        ctx!.lineWidth = 2;
        ctx!.stroke();
      }

      // ── Draw data source nodes ──────────────────────────────────
      const pulseT = Math.sin(time * 0.001 * nodePulseSpeed) * 0.5 + 0.5;

      for (const node of DATA_NODES) {
        const x = lonToX(node.lon, w);
        const y = latToY(node.lat, h);
        const baseSize = node.primary ? 4 : 2.5;
        const pulseSize = baseSize + pulseT * 2 * nodeGlow;

        // Outer glow
        const glowR = pulseSize * 3;
        const grd = ctx!.createRadialGradient(x, y, 0, x, y, glowR);
        const glowAlpha = (0.15 + pulseT * 0.15) * nodeGlow;
        grd.addColorStop(0, `rgba(0, 229, 255, ${Math.min(glowAlpha, 0.6)})`);
        grd.addColorStop(1, "rgba(0, 229, 255, 0)");
        ctx!.fillStyle = grd;
        ctx!.fillRect(x - glowR, y - glowR, glowR * 2, glowR * 2);

        // Core
        ctx!.beginPath();
        ctx!.arc(x, y, pulseSize, 0, Math.PI * 2);
        ctx!.fillStyle = ACCENT;
        ctx!.fill();

        // Label
        ctx!.font = "8px 'JetBrains Mono', monospace";
        ctx!.fillStyle = `rgba(0, 229, 255, ${0.4 + pulseT * 0.2})`;
        ctx!.textAlign = "center";
        ctx!.fillText(node.label, x, y + pulseSize + 10);
      }

      // ── Loading center convergence effect ───────────────────────
      if (state === "loading") {
        const cx = w / 2;
        const cy = h / 2;
        const pulseRadius = 20 + Math.sin(time * 0.005) * 10;
        const grd = ctx!.createRadialGradient(cx, cy, 0, cx, cy, pulseRadius);
        grd.addColorStop(0, "rgba(0, 229, 255, 0.15)");
        grd.addColorStop(1, "rgba(0, 229, 255, 0)");
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
