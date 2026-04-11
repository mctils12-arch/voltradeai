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
// Detailed continent polyline outlines (lat/lon pairs)
// Each array is drawn as connected line segments via ctx.lineTo()
// ────────────────────────────────────────────────────────────────────────────

const CONTINENT_OUTLINES: [number, number][][] = [
  // ── North America (130+ points) ─────────────────────────────────────
  [
    [-168, 65], [-162, 64], [-155, 60], [-150, 61], [-147, 62],
    [-142, 60], [-138, 59], [-136, 58], [-135, 56], [-133, 55],
    [-131, 54], [-130, 53], [-129, 52], [-127, 51], [-125, 49],
    [-124, 48], [-124, 46], [-124, 44], [-123, 42], [-121, 38],
    [-120, 36], [-118, 34], [-117, 33], [-117, 32], [-115, 30],
    [-113, 29], [-111, 27], [-109, 26], [-107, 25], [-105, 23],
    [-103, 21], [-101, 19], [-99, 18], [-97, 17], [-96, 16],
    [-94, 16], [-92, 15], [-90, 15], [-88, 15], [-86, 14],
    [-84, 11], [-83, 10], [-82, 9], [-80, 8], [-79, 9],
    [-78, 9], [-77, 8], [-79, 10], [-81, 11], [-83, 12],
    [-84, 14], [-85, 15], [-87, 16], [-88, 18], [-90, 20],
    [-91, 21], [-90, 22], [-89, 22], [-88, 24], [-87, 25],
    [-85, 27], [-84, 28], [-83, 29], [-82, 30], [-81, 31],
    [-80, 31], [-80, 32], [-79, 34], [-78, 35], [-76, 37],
    [-75, 38], [-74, 40], [-73, 41], [-72, 41], [-71, 42],
    [-70, 42], [-69, 43], [-68, 44], [-67, 45], [-66, 44],
    [-65, 44], [-64, 45], [-63, 46], [-60, 47], [-58, 48],
    [-56, 49], [-55, 50], [-55, 51], [-56, 52], [-59, 53],
    [-60, 54], [-62, 54], [-64, 56], [-66, 58], [-68, 59],
    [-70, 59], [-73, 60], [-75, 61], [-78, 62], [-80, 62],
    [-83, 63], [-85, 64], [-88, 64], [-90, 65], [-93, 66],
    [-95, 67], [-93, 68], [-90, 68], [-88, 69], [-85, 70],
    [-80, 72], [-78, 73], [-75, 72], [-70, 70], [-68, 69],
    [-65, 68], [-62, 67], [-60, 66], [-62, 65], [-65, 64],
    [-68, 63], [-72, 62], [-78, 60], [-82, 58], [-86, 56],
    [-90, 55], [-94, 54], [-98, 54], [-103, 55], [-108, 56],
    [-113, 58], [-118, 58], [-122, 59], [-128, 60], [-133, 60],
    [-138, 60], [-142, 60], [-148, 60], [-153, 58], [-157, 58],
    [-160, 60], [-163, 61], [-166, 63], [-168, 65],
  ],
  // ── Alaska peninsula (separate line) ────────────────────────────────
  [
    [-168, 54], [-166, 55], [-164, 56], [-162, 57], [-160, 58],
    [-157, 57], [-155, 56], [-153, 55], [-155, 54], [-158, 53],
    [-160, 53], [-163, 52], [-165, 52], [-168, 53], [-170, 54],
    [-172, 55], [-174, 56], [-176, 57], [-178, 57],
  ],
  // ── Greenland ───────────────────────────────────────────────────────
  [
    [-52, 60], [-48, 61], [-45, 62], [-43, 63], [-41, 64],
    [-38, 65], [-35, 66], [-30, 68], [-25, 70], [-22, 72],
    [-20, 74], [-18, 76], [-20, 78], [-23, 80], [-28, 81],
    [-33, 82], [-40, 83], [-48, 82], [-53, 81], [-57, 79],
    [-60, 77], [-62, 76], [-63, 74], [-62, 72], [-60, 70],
    [-57, 68], [-55, 66], [-53, 64], [-52, 62], [-52, 60],
  ],
  // ── South America (90+ points) ──────────────────────────────────────
  [
    [-77, 8], [-75, 6], [-73, 5], [-72, 4], [-70, 4],
    [-68, 5], [-66, 6], [-64, 7], [-62, 8], [-60, 7],
    [-58, 6], [-56, 5], [-54, 4], [-52, 3], [-50, 2],
    [-48, 0], [-46, -1], [-44, -2], [-42, -3], [-40, -5],
    [-38, -8], [-36, -10], [-35, -12], [-36, -14], [-37, -16],
    [-38, -18], [-39, -20], [-40, -22], [-42, -23], [-44, -24],
    [-46, -24], [-48, -26], [-48, -28], [-50, -30], [-51, -32],
    [-52, -33], [-53, -34], [-53, -36], [-52, -38], [-50, -40],
    [-52, -42], [-55, -44], [-58, -46], [-62, -48], [-65, -50],
    [-67, -52], [-68, -54], [-70, -53], [-72, -52], [-73, -50],
    [-74, -48], [-73, -46], [-72, -44], [-71, -42], [-71, -40],
    [-71, -38], [-70, -35], [-70, -32], [-70, -30], [-69, -28],
    [-70, -26], [-70, -24], [-70, -22], [-70, -20], [-70, -18],
    [-71, -16], [-73, -14], [-75, -12], [-76, -10], [-77, -8],
    [-78, -6], [-79, -4], [-80, -2], [-80, 0], [-79, 2],
    [-78, 4], [-77, 6], [-77, 8],
  ],
  // ── Europe mainland (110+ points) ───────────────────────────────────
  [
    [-10, 36], [-8, 37], [-6, 37], [-5, 36], [-2, 36],
    [0, 38], [1, 39], [2, 41], [3, 43], [5, 43],
    [6, 44], [8, 44], [9, 43], [10, 44], [12, 44],
    [13, 43], [14, 42], [15, 42], [16, 41], [17, 40],
    [18, 40], [19, 39], [20, 38], [21, 38], [22, 37],
    [23, 37], [24, 36], [26, 36], [28, 37], [29, 38],
    [30, 39], [31, 40], [32, 41], [33, 42], [35, 42],
    [37, 42], [38, 43], [40, 44], [40, 46], [38, 47],
    [36, 46], [34, 46], [32, 46], [30, 45], [29, 46],
    [28, 46], [27, 47], [25, 48], [22, 48], [20, 48],
    [18, 48], [17, 49], [16, 50], [15, 51], [14, 52],
    [14, 54], [13, 55], [12, 55], [11, 56], [10, 56],
    [10, 57], [11, 58], [12, 59], [13, 60], [14, 61],
    [16, 62], [18, 63], [20, 64], [22, 65], [24, 66],
    [26, 68], [28, 70], [30, 70], [28, 71], [24, 71],
    [20, 70], [18, 69], [16, 68], [14, 67], [12, 66],
    [10, 64], [8, 62], [6, 60], [5, 58], [6, 56],
    [8, 55], [8, 54], [6, 53], [4, 52], [3, 51],
    [2, 50], [0, 50], [-2, 50], [-4, 48], [-5, 47],
    [-4, 46], [-2, 44], [-1, 43], [-2, 42], [-4, 41],
    [-5, 40], [-7, 39], [-8, 38], [-9, 37], [-10, 36],
  ],
  // ── British Isles ───────────────────────────────────────────────────
  [
    [-6, 50], [-5, 51], [-4, 52], [-3, 53], [-4, 54],
    [-5, 55], [-5, 56], [-4, 57], [-3, 58], [-2, 58],
    [-1, 58], [0, 57], [1, 55], [2, 53], [1, 52],
    [0, 51], [-1, 50], [-3, 50], [-5, 50], [-6, 50],
  ],
  // ── Ireland ─────────────────────────────────────────────────────────
  [
    [-10, 52], [-9, 53], [-8, 54], [-7, 55], [-8, 55],
    [-10, 54], [-10, 53], [-10, 52],
  ],
  // ── Iceland ─────────────────────────────────────────────────────────
  [
    [-24, 64], [-22, 64], [-20, 65], [-18, 66], [-16, 66],
    [-14, 65], [-14, 64], [-16, 63], [-18, 63], [-20, 63],
    [-22, 63], [-24, 64],
  ],
  // ── Scandinavian peninsula ──────────────────────────────────────────
  [
    [5, 58], [6, 60], [7, 62], [10, 64], [14, 66],
    [16, 68], [18, 69], [20, 70], [24, 71], [28, 70],
    [30, 70], [30, 68], [28, 66], [25, 64], [22, 62],
    [20, 60], [18, 58], [16, 57], [14, 56], [12, 56],
    [10, 57], [8, 58], [5, 58],
  ],
  // ── Africa (90+ points) ─────────────────────────────────────────────
  [
    [-17, 15], [-16, 17], [-16, 19], [-17, 21], [-16, 23],
    [-14, 25], [-13, 27], [-11, 29], [-9, 31], [-7, 33],
    [-5, 34], [-3, 35], [-1, 35], [1, 36], [3, 37],
    [5, 37], [8, 37], [10, 37], [11, 35], [10, 33],
    [10, 31], [12, 30], [14, 30], [16, 30], [18, 30],
    [20, 31], [22, 31], [24, 31], [25, 30], [28, 29],
    [30, 28], [32, 27], [34, 26], [36, 24], [38, 22],
    [40, 18], [42, 16], [44, 12], [46, 10], [48, 8],
    [50, 10], [51, 12], [50, 14], [48, 12], [46, 10],
    [44, 8], [42, 5], [42, 2], [41, 0], [40, -2],
    [40, -5], [39, -8], [38, -10], [37, -12], [36, -14],
    [35, -18], [35, -22], [34, -26], [32, -28], [30, -30],
    [28, -32], [27, -34], [25, -34], [22, -34], [20, -33],
    [18, -30], [17, -25], [16, -20], [14, -15], [12, -10],
    [10, -5], [10, -2], [9, 0], [8, 2], [6, 4],
    [4, 4], [2, 4], [0, 4], [-2, 4], [-4, 5],
    [-6, 5], [-8, 5], [-10, 5], [-12, 6], [-14, 8],
    [-16, 10], [-17, 12], [-17, 15],
  ],
  // ── Madagascar ──────────────────────────────────────────────────────
  [
    [44, -12], [46, -14], [48, -16], [49, -18], [50, -20],
    [49, -22], [48, -24], [46, -25], [44, -24], [43, -22],
    [43, -20], [43, -18], [43, -16], [44, -14], [44, -12],
  ],
  // ── Asia (130+ points) ──────────────────────────────────────────────
  [
    [28, 42], [30, 42], [32, 38], [34, 36], [36, 34],
    [38, 32], [40, 30], [42, 28], [44, 26], [46, 24],
    [48, 22], [50, 22], [52, 24], [54, 25], [56, 26],
    [58, 25], [60, 24], [62, 25], [64, 26], [66, 25],
    [68, 24], [70, 22], [72, 20], [74, 18], [76, 14],
    [78, 10], [80, 8], [80, 10], [80, 14], [79, 16],
    [78, 18], [76, 20], [74, 22], [72, 24], [72, 28],
    [72, 32], [70, 34], [68, 36], [66, 38], [64, 40],
    [62, 42], [60, 44], [58, 46], [56, 48], [54, 50],
    [56, 52], [58, 54], [60, 56], [62, 58], [64, 60],
    [68, 62], [72, 64], [76, 66], [80, 68], [85, 70],
    [90, 72], [95, 72], [100, 72], [110, 72], [120, 72],
    [130, 70], [140, 68], [150, 66], [155, 64], [160, 62],
    [165, 60], [168, 58], [170, 56], [168, 54], [165, 52],
    [162, 50], [158, 48], [155, 46], [150, 44], [148, 42],
    [145, 42], [142, 44], [140, 46], [138, 44], [135, 42],
    [133, 40], [130, 38], [128, 35], [126, 34], [125, 36],
    [124, 38], [122, 38], [120, 36], [118, 34], [116, 32],
    [114, 30], [112, 28], [110, 24], [108, 20], [106, 18],
    [104, 16], [102, 14], [100, 12], [100, 10], [102, 8],
    [104, 4], [104, 2], [102, 1], [100, 2], [98, 4],
    [98, 8], [98, 12], [96, 16], [94, 20], [92, 22],
    [90, 22], [88, 22], [86, 24], [84, 26], [82, 28],
    [80, 30], [78, 32], [76, 34], [74, 34], [72, 34],
    [70, 36], [68, 38], [66, 40], [64, 42], [60, 44],
    [55, 44], [50, 44], [45, 42], [40, 42], [35, 42],
    [32, 42], [28, 42],
  ],
  // ── India subcontinent detail ───────────────────────────────────────
  [
    [68, 24], [70, 22], [72, 20], [74, 18], [76, 14],
    [78, 10], [80, 8], [80, 10], [80, 14], [79, 16],
    [78, 18], [76, 20], [74, 22], [72, 24], [70, 24],
    [68, 24],
  ],
  // ── Japan ───────────────────────────────────────────────────────────
  [
    [130, 31], [131, 33], [133, 34], [135, 35], [137, 36],
    [139, 37], [140, 38], [141, 40], [142, 42], [143, 44],
    [145, 45], [145, 44], [143, 42], [142, 40], [141, 38],
    [140, 36], [138, 34], [136, 33], [134, 32], [132, 31],
    [130, 31],
  ],
  // ── Korean peninsula ────────────────────────────────────────────────
  [
    [126, 34], [127, 35], [128, 36], [129, 38], [128, 40],
    [127, 42], [125, 42], [124, 40], [125, 38], [125, 36],
    [126, 34],
  ],
  // ── Southeast Asia / Malay peninsula ────────────────────────────────
  [
    [100, 12], [101, 10], [102, 8], [103, 5], [104, 2],
    [104, 1], [103, 2], [102, 4], [101, 6], [100, 8],
    [100, 10], [100, 12],
  ],
  // ── Indonesia (Sumatra + Java + Borneo arcs) ────────────────────────
  [
    [95, 5], [97, 3], [99, 1], [101, 0], [103, -1],
    [105, -3], [106, -5], [108, -6], [110, -7], [112, -8],
    [114, -8], [116, -8], [118, -8], [120, -8], [122, -7],
    [124, -6], [126, -5], [128, -4], [130, -3], [132, -2],
    [134, -2], [136, -3], [138, -4], [140, -5], [141, -6],
  ],
  // ── Borneo ──────────────────────────────────────────────────────────
  [
    [109, 1], [110, 2], [112, 3], [114, 4], [116, 4],
    [118, 3], [119, 2], [118, 0], [117, -1], [115, -2],
    [113, -2], [111, -1], [110, 0], [109, 1],
  ],
  // ── Philippines ─────────────────────────────────────────────────────
  [
    [118, 5], [119, 7], [120, 10], [121, 12], [122, 14],
    [122, 16], [121, 18], [120, 18], [119, 16], [118, 14],
    [117, 12], [117, 10], [117, 8], [118, 5],
  ],
  // ── Australia (60+ points) ──────────────────────────────────────────
  [
    [114, -22], [114, -24], [114, -26], [115, -28],
    [115, -32], [116, -34], [118, -35], [120, -35],
    [122, -34], [124, -34], [126, -33], [128, -32],
    [130, -31], [131, -32], [132, -33], [134, -34],
    [136, -35], [138, -36], [140, -38], [142, -38],
    [144, -38], [146, -39], [148, -38], [150, -37],
    [151, -34], [153, -30], [153, -28], [153, -26],
    [152, -24], [150, -22], [148, -20], [146, -18],
    [144, -16], [142, -14], [140, -12], [138, -12],
    [136, -12], [134, -12], [132, -12], [130, -12],
    [128, -14], [126, -14], [124, -14], [122, -14],
    [120, -14], [118, -16], [116, -18], [114, -20],
    [114, -22],
  ],
  // ── Tasmania ────────────────────────────────────────────────────────
  [
    [144, -40], [146, -41], [148, -42], [148, -43],
    [146, -44], [144, -43], [144, -42], [144, -40],
  ],
  // ── New Zealand (North Island) ──────────────────────────────────────
  [
    [172, -34], [174, -36], [176, -37], [178, -38],
    [178, -40], [176, -41], [174, -40], [173, -38],
    [172, -36], [172, -34],
  ],
  // ── New Zealand (South Island) ──────────────────────────────────────
  [
    [168, -42], [170, -42], [172, -43], [174, -44],
    [172, -46], [170, -46], [168, -45], [167, -44],
    [168, -42],
  ],
  // ── Sri Lanka ───────────────────────────────────────────────────────
  [
    [80, 10], [81, 8], [82, 7], [82, 6], [81, 6],
    [80, 7], [79, 8], [80, 10],
  ],
  // ── Antarctica (simple outline at bottom) ───────────────────────────
  [
    [-180, -60], [-160, -62], [-140, -64], [-120, -66],
    [-100, -68], [-80, -70], [-60, -68], [-40, -66],
    [-20, -64], [0, -62], [20, -64], [40, -66],
    [60, -68], [80, -70], [100, -68], [120, -66],
    [140, -64], [160, -62], [180, -60],
  ],
];

// ────────────────────────────────────────────────────────────────────────────
// Projection helpers — full page edge-to-edge
// lon: -180..180 → 0..width
// lat:  ~80..-60 → 0..height (simple equirectangular stretched)
// ────────────────────────────────────────────────────────────────────────────

const LAT_MAX = 80;
const LAT_MIN = -60;

function lonToX(lon: number, w: number): number {
  return ((lon + 180) / 360) * w;
}

function latToY(lat: number, h: number): number {
  const clamped = Math.max(LAT_MIN, Math.min(LAT_MAX, lat));
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
const USER_COLOR = "#ffd700";
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

      // ── Draw continent outlines ───────────────────────────────────
      for (const outline of CONTINENT_OUTLINES) {
        if (outline.length < 2) continue;

        // Subtle fill
        ctx!.beginPath();
        ctx!.moveTo(lonToX(outline[0][0], w), latToY(outline[0][1], h));
        for (let i = 1; i < outline.length; i++) {
          ctx!.lineTo(lonToX(outline[i][0], w), latToY(outline[i][1], h));
        }
        ctx!.closePath();
        ctx!.fillStyle = "rgba(0, 229, 255, 0.03)";
        ctx!.fill();

        // Stroke outline
        ctx!.beginPath();
        ctx!.moveTo(lonToX(outline[0][0], w), latToY(outline[0][1], h));
        for (let i = 1; i < outline.length; i++) {
          ctx!.lineTo(lonToX(outline[i][0], w), latToY(outline[i][1], h));
        }
        ctx!.strokeStyle = "rgba(0, 229, 255, 0.12)";
        ctx!.lineWidth = 0.6;
        ctx!.stroke();
      }

      // ── Draw connection base lines ────────────────────────────────
      for (const [fromIdx, toIdx] of connections) {
        if (fromIdx >= nodes.length || toIdx >= nodes.length) continue;
        const from = nodes[fromIdx];
        const to = nodes[toIdx];
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
        ctx!.strokeStyle = toIdx === USER_NODE_IDX
          ? `rgba(255, 215, 0, ${lineAlpha})`
          : `rgba(0, 229, 255, ${lineAlpha})`;
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
        const x1 = lonToX(from.lon, w);
        const y1 = latToY(from.lat, h);
        const x2 = lonToX(to.lon, w);
        const y2 = latToY(to.lat, h);

        const pos = arcPoint(x1, y1, x2, y2, p.t, w);

        // Glow
        const glowRadius = (p.size + 3) * (state === "loading" ? 1.5 : 1);
        const gradient = ctx!.createRadialGradient(pos.x, pos.y, 0, pos.x, pos.y, glowRadius);
        const isUserPath = p.toIdx === USER_NODE_IDX;
        if (isUserPath) {
          gradient.addColorStop(0, `rgba(255, 215, 0, ${p.opacity * 0.5})`);
          gradient.addColorStop(1, "rgba(255, 215, 0, 0)");
        } else {
          gradient.addColorStop(0, `rgba(0, 229, 255, ${p.opacity * 0.6})`);
          gradient.addColorStop(1, "rgba(0, 229, 255, 0)");
        }
        ctx!.fillStyle = gradient;
        ctx!.fillRect(pos.x - glowRadius, pos.y - glowRadius, glowRadius * 2, glowRadius * 2);

        // Core dot
        ctx!.beginPath();
        ctx!.arc(pos.x, pos.y, p.size * 0.6, 0, Math.PI * 2);
        ctx!.fillStyle = isUserPath
          ? `rgba(255, 215, 0, ${p.opacity})`
          : `rgba(0, 229, 255, ${p.opacity})`;
        ctx!.fill();
      }

      // ── Draw burst ripple ─────────────────────────────────────────
      if (inBurst && nodes.length > USER_NODE_IDX) {
        const userNode = nodes[USER_NODE_IDX];
        const centerX = lonToX(userNode.lon, w);
        const centerY = latToY(userNode.lat, h);
        const rippleRadius = burstAge * 400;
        const rippleAlpha = Math.max(0, 0.3 * (1 - burstAge / 1.5));
        ctx!.beginPath();
        ctx!.arc(centerX, centerY, rippleRadius, 0, Math.PI * 2);
        ctx!.strokeStyle = `rgba(255, 215, 0, ${rippleAlpha})`;
        ctx!.lineWidth = 2;
        ctx!.stroke();
      }

      // ── Draw data source nodes ────────────────────────────────────
      const pulseT = Math.sin(time * 0.001 * nodePulseSpeed) * 0.5 + 0.5;

      for (let n = 0; n < nodes.length; n++) {
        const node = nodes[n];
        const x = lonToX(node.lon, w);
        const y = latToY(node.lat, h);
        const isUser = n === USER_NODE_IDX;

        const baseSize = isUser ? 5 : node.primary ? 4 : 2.5;
        const pulseSize = baseSize + pulseT * 2 * nodeGlow;

        // Outer glow
        const glowR = pulseSize * (isUser ? 4 : 3);
        const grd = ctx!.createRadialGradient(x, y, 0, x, y, glowR);
        const glowAlpha = (0.15 + pulseT * 0.15) * nodeGlow;
        if (isUser) {
          grd.addColorStop(0, `rgba(255, 215, 0, ${Math.min(glowAlpha, 0.6)})`);
          grd.addColorStop(1, "rgba(255, 215, 0, 0)");
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
        ctx!.font = isUser ? "bold 9px 'JetBrains Mono', monospace" : "8px 'JetBrains Mono', monospace";
        ctx!.fillStyle = isUser
          ? `rgba(255, 215, 0, ${0.6 + pulseT * 0.3})`
          : `rgba(0, 229, 255, ${0.4 + pulseT * 0.2})`;
        ctx!.textAlign = "center";
        ctx!.fillText(node.label, x, y + pulseSize + 10);
      }

      // ── Loading center convergence effect ─────────────────────────
      if (state === "loading" && nodes.length > USER_NODE_IDX) {
        const userNode = nodes[USER_NODE_IDX];
        const cx = lonToX(userNode.lon, w);
        const cy = latToY(userNode.lat, h);
        const pulseRadius = 20 + Math.sin(time * 0.005) * 10;
        const grd = ctx!.createRadialGradient(cx, cy, 0, cx, cy, pulseRadius);
        grd.addColorStop(0, "rgba(255, 215, 0, 0.15)");
        grd.addColorStop(1, "rgba(255, 215, 0, 0)");
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
