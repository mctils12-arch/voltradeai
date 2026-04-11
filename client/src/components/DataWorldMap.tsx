import { useRef, useEffect, useCallback } from "react";
import * as THREE from "three";
import { CONTINENT_OUTLINES, type LatLon } from "./worldGeoData";

// ────────────────────────────────────────────────────────────────────────────
// Types
// ────────────────────────────────────────────────────────────────────────────

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

type AnimState = "idle" | "loading" | "loaded";

// ────────────────────────────────────────────────────────────────────────────
// Data
// ────────────────────────────────────────────────────────────────────────────

const DATA_NODES: CityNode[] = [
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

const CONNECTIONS: [number, number][] = [
  [1, 0], [2, 0], [3, 0], [4, 3], [5, 0],
  [6, 5], [7, 6], [8, 3], [9, 0], [10, 6],
  [11, 0], [4, 0], [8, 0], [5, 2], [10, 2],
];

const GLOBE_RADIUS = 2;
const ACCENT_COLOR = new THREE.Color("#00e5ff");
const DEG2RAD = Math.PI / 180;
const AXIAL_TILT = 23.5 * DEG2RAD;

// ────────────────────────────────────────────────────────────────────────────
// Helpers
// ────────────────────────────────────────────────────────────────────────────

function latLonToVec3(lat: number, lon: number, radius: number): THREE.Vector3 {
  const phi = (90 - lat) * DEG2RAD;
  const theta = (lon + 180) * DEG2RAD;
  return new THREE.Vector3(
    -radius * Math.sin(phi) * Math.cos(theta),
    radius * Math.cos(phi),
    radius * Math.sin(phi) * Math.sin(theta),
  );
}

function createGlowTexture(size: number, color: [number, number, number]): THREE.Texture {
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d")!;
  const gradient = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
  gradient.addColorStop(0, `rgba(${color[0]}, ${color[1]}, ${color[2]}, 1)`);
  gradient.addColorStop(0.3, `rgba(${color[0]}, ${color[1]}, ${color[2]}, 0.5)`);
  gradient.addColorStop(0.7, `rgba(${color[0]}, ${color[1]}, ${color[2]}, 0.1)`);
  gradient.addColorStop(1, `rgba(${color[0]}, ${color[1]}, ${color[2]}, 0)`);
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, size, size);
  const texture = new THREE.CanvasTexture(canvas);
  return texture;
}

function createLabelTexture(text: string): THREE.Texture {
  const canvas = document.createElement("canvas");
  canvas.width = 64;
  canvas.height = 32;
  const ctx = canvas.getContext("2d")!;
  ctx.clearRect(0, 0, 64, 32);
  ctx.font = "bold 16px 'JetBrains Mono', monospace";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillStyle = "rgba(0, 229, 255, 0.6)";
  ctx.fillText(text, 32, 16);
  const texture = new THREE.CanvasTexture(canvas);
  return texture;
}

// ────────────────────────────────────────────────────────────────────────────
// Arc geometry helpers
// ────────────────────────────────────────────────────────────────────────────

function createArcCurve(from: CityNode, to: CityNode): THREE.QuadraticBezierCurve3 {
  const start = latLonToVec3(from.lat, from.lon, GLOBE_RADIUS + 0.01);
  const end = latLonToVec3(to.lat, to.lon, GLOBE_RADIUS + 0.01);
  const mid = new THREE.Vector3().addVectors(start, end).multiplyScalar(0.5);
  const dist = start.distanceTo(end);
  const altitude = GLOBE_RADIUS + 0.3 + dist * 0.15;
  mid.normalize().multiplyScalar(altitude);
  return new THREE.QuadraticBezierCurve3(start, mid, end);
}

// ────────────────────────────────────────────────────────────────────────────
// Component
// ────────────────────────────────────────────────────────────────────────────

export default function DataWorldMap({ isLoading, hasData, ticker }: DataWorldMapProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const stateRef = useRef<AnimState>("idle");
  const prevStateRef = useRef<AnimState>("idle");
  const burstTimeRef = useRef(0);
  const propsRef = useRef({ isLoading, hasData, ticker });

  // Keep props in a ref so the animation loop always reads the latest values
  useEffect(() => {
    propsRef.current = { isLoading, hasData, ticker };
  }, [isLoading, hasData, ticker]);

  const getState = useCallback((): AnimState => {
    const { isLoading: l, hasData: d, ticker: t } = propsRef.current;
    if (l) return "loading";
    if (d && t) return "loaded";
    return "idle";
  }, []);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    // ── Renderer ──────────────────────────────────────────────────────
    const renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: true,
      powerPreference: "low-power",
    });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(window.innerWidth, window.innerHeight);
    container.appendChild(renderer.domElement);

    // ── Scene & Camera ───────────────────────────────────────────────
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(
      45,
      window.innerWidth / window.innerHeight,
      0.1,
      100,
    );
    camera.position.set(0, 0.5, 4.5);
    camera.lookAt(0, 0, 0);

    // ── Lighting ─────────────────────────────────────────────────────
    const ambient = new THREE.AmbientLight(0x222244, 0.5);
    scene.add(ambient);
    const directional = new THREE.DirectionalLight(0xffffff, 0.3);
    directional.position.set(3, 4, 2);
    scene.add(directional);

    // ── Globe group (tilted) ─────────────────────────────────────────
    const globeGroup = new THREE.Group();
    globeGroup.rotation.x = AXIAL_TILT;
    scene.add(globeGroup);

    // ── Inner sphere ─────────────────────────────────────────────────
    const innerGeo = new THREE.SphereGeometry(GLOBE_RADIUS - 0.01, 48, 48);
    const innerMat = new THREE.MeshBasicMaterial({
      color: 0x050a12,
      transparent: true,
      opacity: 0.7,
    });
    const innerSphere = new THREE.Mesh(innerGeo, innerMat);
    globeGroup.add(innerSphere);

    // ── Outer wireframe grid (lat/lon lines) ─────────────────────────
    const gridGroup = new THREE.Group();
    const gridMat = new THREE.LineBasicMaterial({
      color: ACCENT_COLOR,
      transparent: true,
      opacity: 0.06,
    });

    // Latitude lines
    for (let i = 0; i < 24; i++) {
      const lat = -90 + (180 / 24) * i;
      const phi = (90 - lat) * DEG2RAD;
      const points: THREE.Vector3[] = [];
      for (let j = 0; j <= 72; j++) {
        const theta = (j / 72) * Math.PI * 2;
        points.push(new THREE.Vector3(
          -GLOBE_RADIUS * Math.sin(phi) * Math.cos(theta),
          GLOBE_RADIUS * Math.cos(phi),
          GLOBE_RADIUS * Math.sin(phi) * Math.sin(theta),
        ));
      }
      const geo = new THREE.BufferGeometry().setFromPoints(points);
      gridGroup.add(new THREE.Line(geo, gridMat));
    }

    // Longitude lines
    for (let i = 0; i < 36; i++) {
      const lon = -180 + (360 / 36) * i;
      const theta = (lon + 180) * DEG2RAD;
      const points: THREE.Vector3[] = [];
      for (let j = 0; j <= 72; j++) {
        const phi = (j / 72) * Math.PI;
        points.push(new THREE.Vector3(
          -GLOBE_RADIUS * Math.sin(phi) * Math.cos(theta),
          GLOBE_RADIUS * Math.cos(phi),
          GLOBE_RADIUS * Math.sin(phi) * Math.sin(theta),
        ));
      }
      const geo = new THREE.BufferGeometry().setFromPoints(points);
      gridGroup.add(new THREE.Line(geo, gridMat));
    }
    globeGroup.add(gridGroup);

    // ── Continent outlines ───────────────────────────────────────────
    const continentMat = new THREE.LineBasicMaterial({
      color: ACCENT_COLOR,
      transparent: true,
      opacity: 0.25,
    });

    for (const outline of CONTINENT_OUTLINES) {
      const points = outline.map(([lat, lon]: LatLon) =>
        latLonToVec3(lat, lon, GLOBE_RADIUS + 0.012),
      );
      if (points.length < 2) continue;
      const geo = new THREE.BufferGeometry().setFromPoints(points);
      const line = new THREE.Line(geo, continentMat);
      globeGroup.add(line);
    }

    // ── City nodes (sprites) ─────────────────────────────────────────
    const glowTex = createGlowTexture(64, [0, 229, 255]);
    const nodeSprites: THREE.Sprite[] = [];
    const labelSprites: THREE.Sprite[] = [];

    for (const node of DATA_NODES) {
      const pos = latLonToVec3(node.lat, node.lon, GLOBE_RADIUS + 0.03);

      // Glow sprite
      const spriteMat = new THREE.SpriteMaterial({
        map: glowTex,
        transparent: true,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
      });
      const sprite = new THREE.Sprite(spriteMat);
      sprite.position.copy(pos);
      const baseScale = node.primary ? 0.18 : 0.12;
      sprite.scale.set(baseScale, baseScale, 1);
      sprite.userData = { baseScale, primary: !!node.primary };
      globeGroup.add(sprite);
      nodeSprites.push(sprite);

      // Label
      const labelTex = createLabelTexture(node.label);
      const labelMat = new THREE.SpriteMaterial({
        map: labelTex,
        transparent: true,
        depthWrite: false,
      });
      const labelSprite = new THREE.Sprite(labelMat);
      const labelOffset = pos.clone().normalize().multiplyScalar(0.15);
      labelSprite.position.copy(pos).add(labelOffset);
      labelSprite.scale.set(0.25, 0.12, 1);
      globeGroup.add(labelSprite);
      labelSprites.push(labelSprite);
    }

    // ── Arc lines ────────────────────────────────────────────────────
    const arcCurves: THREE.QuadraticBezierCurve3[] = [];
    const arcLines: THREE.Line[] = [];
    const arcMats: THREE.LineBasicMaterial[] = [];

    for (const [fromIdx, toIdx] of CONNECTIONS) {
      const curve = createArcCurve(DATA_NODES[fromIdx], DATA_NODES[toIdx]);
      arcCurves.push(curve);
      const points = curve.getPoints(50);
      const geo = new THREE.BufferGeometry().setFromPoints(points);
      const mat = new THREE.LineBasicMaterial({
        color: ACCENT_COLOR,
        transparent: true,
        opacity: 0.08,
      });
      arcMats.push(mat);
      const line = new THREE.Line(geo, mat);
      globeGroup.add(line);
      arcLines.push(line);
    }

    // ── Particle system ──────────────────────────────────────────────
    const MAX_PARTICLES = 200;
    const particlePositions = new Float32Array(MAX_PARTICLES * 3);
    const particleSizes = new Float32Array(MAX_PARTICLES);
    const particleOpacities = new Float32Array(MAX_PARTICLES);

    interface ParticleData {
      arcIdx: number;
      t: number;
      speed: number;
      active: boolean;
      trailOffset: number; // 0 = lead, 1/2 = trail
    }

    const particleData: ParticleData[] = [];
    for (let i = 0; i < MAX_PARTICLES; i++) {
      particleData.push({
        arcIdx: Math.floor(Math.random() * CONNECTIONS.length),
        t: Math.random(),
        speed: 0.15 + Math.random() * 0.25,
        active: i < 40,
        trailOffset: i % 3 === 0 ? 0 : i % 3 === 1 ? 0.02 : 0.04,
      });
    }

    const particleGeo = new THREE.BufferGeometry();
    particleGeo.setAttribute("position", new THREE.BufferAttribute(particlePositions, 3));
    particleGeo.setAttribute("size", new THREE.BufferAttribute(particleSizes, 1));

    const particleTex = createGlowTexture(32, [0, 229, 255]);
    const particleMat = new THREE.PointsMaterial({
      map: particleTex,
      size: 0.06,
      transparent: true,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      sizeAttenuation: true,
    });

    const particleSystem = new THREE.Points(particleGeo, particleMat);
    globeGroup.add(particleSystem);

    // ── Energy ring (visible during loading) ─────────────────────────
    const ringGeo = new THREE.RingGeometry(GLOBE_RADIUS + 0.05, GLOBE_RADIUS + 0.08, 64);
    const ringMat = new THREE.MeshBasicMaterial({
      color: ACCENT_COLOR,
      transparent: true,
      opacity: 0,
      side: THREE.DoubleSide,
      blending: THREE.AdditiveBlending,
    });
    const energyRing = new THREE.Mesh(ringGeo, ringMat);
    energyRing.rotation.x = Math.PI / 2;
    globeGroup.add(energyRing);

    // ── Track disposables ────────────────────────────────────────────
    const disposables: { dispose: () => void }[] = [
      innerGeo, innerMat, gridMat, continentMat, glowTex,
      particleGeo, particleMat, particleTex, ringGeo, ringMat,
    ];
    for (const sprite of nodeSprites) disposables.push(sprite.material as THREE.SpriteMaterial);
    for (const sprite of labelSprites) {
      const mat = sprite.material as THREE.SpriteMaterial;
      if (mat.map) disposables.push(mat.map);
      disposables.push(mat);
    }
    for (const line of arcLines) disposables.push(line.geometry);
    for (const mat of arcMats) disposables.push(mat);
    gridGroup.children.forEach((child) => {
      if (child instanceof THREE.Line) disposables.push(child.geometry);
    });

    // ── Animation ────────────────────────────────────────────────────
    const clock = new THREE.Clock();
    let animId = 0;
    let running = true;

    function animate() {
      if (!running) return;
      animId = requestAnimationFrame(animate);

      const dt = clock.getDelta();
      const elapsed = clock.getElapsedTime();

      const newState = getState();
      if (newState === "loaded" && prevStateRef.current === "loading") {
        burstTimeRef.current = elapsed;
      }
      prevStateRef.current = stateRef.current;
      stateRef.current = newState;

      const state = stateRef.current;
      const burstAge = elapsed - burstTimeRef.current;
      const inBurst = burstAge < 0.5 && burstTimeRef.current > 0;

      // ── State-based parameters ─────────────────────────────────
      let rotSpeed = 0.0005;
      let targetActive = 40;
      let speedMult = 1;
      let arcOpacity = 0.08;
      let nodeGlow = 1;
      let nodePulseFreq = 1;
      let ringOpacity = 0;

      if (state === "loading") {
        rotSpeed = 0.002;
        targetActive = 150;
        speedMult = 4;
        arcOpacity = 0.2;
        nodeGlow = 2;
        nodePulseFreq = 3;
        ringOpacity = 0.15;
      } else if (state === "loaded") {
        if (inBurst) {
          rotSpeed = 0.003;
          targetActive = 180;
          speedMult = 6;
          arcOpacity = 0.25;
          nodeGlow = 2.5;
          nodePulseFreq = 4;
        } else {
          rotSpeed = 0.001;
          targetActive = 60;
          speedMult = 1.8;
          arcOpacity = 0.12;
          nodeGlow = 1.3;
          nodePulseFreq = 1.5;
        }
      }

      // ── Rotate globe ───────────────────────────────────────────
      globeGroup.rotation.y += rotSpeed;

      // ── Update arc opacities ───────────────────────────────────
      for (const mat of arcMats) {
        mat.opacity += (arcOpacity - mat.opacity) * 0.05;
      }

      // ── Update energy ring ─────────────────────────────────────
      ringMat.opacity += (ringOpacity - ringMat.opacity) * 0.05;
      energyRing.rotation.z += state === "loading" ? 0.03 : 0.005;

      // ── Update node sprites ────────────────────────────────────
      const pulse = Math.sin(elapsed * nodePulseFreq * 2) * 0.5 + 0.5;
      for (const sprite of nodeSprites) {
        const base: number = sprite.userData.baseScale;
        const scale = base * (1 + pulse * 0.15 * nodeGlow);
        sprite.scale.set(scale, scale, 1);
        (sprite.material as THREE.SpriteMaterial).opacity = 0.5 + pulse * 0.3 * nodeGlow;
      }

      // ── Update particles ───────────────────────────────────────
      let activeCount = 0;
      for (let i = 0; i < MAX_PARTICLES; i++) {
        const p = particleData[i];
        if (activeCount < targetActive && !p.active) {
          p.active = true;
          p.t = 0;
          p.arcIdx = Math.floor(Math.random() * CONNECTIONS.length);
          p.speed = 0.15 + Math.random() * 0.25;
        }
        if (p.active) activeCount++;

        if (!p.active) {
          particlePositions[i * 3] = 0;
          particlePositions[i * 3 + 1] = 0;
          particlePositions[i * 3 + 2] = -999;
          particleSizes[i] = 0;
          continue;
        }

        p.t += p.speed * speedMult * dt;

        // Burst scatter effect
        let scatterOffset = 0;
        if (inBurst && p.trailOffset === 0) {
          scatterOffset = Math.sin(burstAge * Math.PI * 2) * 0.1;
        }

        if (p.t >= 1) {
          if (activeCount > targetActive + 10) {
            p.active = false;
            particlePositions[i * 3 + 2] = -999;
            particleSizes[i] = 0;
            continue;
          }
          p.t = 0;
          p.arcIdx = Math.floor(Math.random() * CONNECTIONS.length);
          p.speed = 0.15 + Math.random() * 0.25;
        }

        const tClamped = Math.max(0, Math.min(1, p.t - p.trailOffset));
        const curve = arcCurves[p.arcIdx];
        const pos = curve.getPoint(tClamped);

        if (scatterOffset > 0) {
          const normal = pos.clone().normalize();
          pos.add(normal.multiplyScalar(scatterOffset));
        }

        particlePositions[i * 3] = pos.x;
        particlePositions[i * 3 + 1] = pos.y;
        particlePositions[i * 3 + 2] = pos.z;

        const trailFade = p.trailOffset === 0 ? 1 : p.trailOffset < 0.03 ? 0.6 : 0.3;
        particleSizes[i] = (0.04 + 0.02 * trailFade) * (state === "loading" ? 1.3 : 1);
      }

      particleGeo.attributes.position.needsUpdate = true;
      particleGeo.attributes.size.needsUpdate = true;

      renderer.render(scene, camera);
    }

    animate();

    // ── Resize handler ───────────────────────────────────────────────
    function onResize() {
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    }
    window.addEventListener("resize", onResize);

    // ── Cleanup ──────────────────────────────────────────────────────
    return () => {
      running = false;
      cancelAnimationFrame(animId);
      window.removeEventListener("resize", onResize);

      for (const d of disposables) {
        d.dispose();
      }
      renderer.dispose();

      if (container.contains(renderer.domElement)) {
        container.removeChild(renderer.domElement);
      }
    };
  }, [getState]);

  return (
    <div
      ref={containerRef}
      style={{
        position: "fixed",
        top: 0,
        left: 0,
        width: "100%",
        height: "100%",
        zIndex: 0,
        pointerEvents: "none",
      }}
    />
  );
}
