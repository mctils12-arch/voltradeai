import { useState, useEffect, useRef } from "react";
import { queryClient } from "@/lib/queryClient";

interface LoginProps {
  onLogin?: () => void;
}

// ── Matrix City Canvas ──────────────────────────────────────────────────────

function CityMatrixCanvas() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let animId: number;
    let W = window.innerWidth;
    let H = window.innerHeight;
    canvas.width = W;
    canvas.height = H;

    const handleResize = () => {
      W = window.innerWidth;
      H = window.innerHeight;
      canvas.width = W;
      canvas.height = H;
      generateCity();
    };
    window.addEventListener("resize", handleResize);

    // ── Types ──
    interface Setback { fromTop: number; insetLeft: number; insetRight: number }
    interface RoofDetail { type: "water-tower" | "mech-penthouse" | "helipad" | "spire" | "art-deco" | "angular-crown" | "stepped-crown"; xOff: number; params: Record<string, number> }
    interface Window { wx: number; wy: number; ww: number; wh: number; lit: boolean; warmth: number; flicker: number }
    interface Building {
      x: number; w: number; h: number; layer: number;
      setbacks: Setback[];
      roofDetails: RoofDetail[];
      antennaHeight: number;
      windows: Window[];
      reflectionX: number; reflectionW: number;
    }
    interface Stream {
      x: number; y: number; speed: number; length: number; chars: string[];
      opacity: number; buildingTop: number;
    }

    const CHARS = "01".split("");
    let layers: Building[][] = [[], [], []]; // back, mid, fore
    let streams: Stream[] = [];

    // ── Building shape generator ──
    function makeBuilding(x: number, baseW: number, baseH: number, layer: number, groundY: number): Building {
      const setbacks: Setback[] = [];
      const roofDetails: RoofDetail[] = [];
      let w = baseW;
      let h = baseH;

      // Taller buildings get setbacks (stepped profile)
      if (h > H * 0.3 && Math.random() < 0.6) {
        const numSetbacks = 1 + Math.floor(Math.random() * 2);
        for (let s = 0; s < numSetbacks; s++) {
          const fromTop = h * (0.15 + s * 0.2 + Math.random() * 0.1);
          const inset = w * (0.08 + Math.random() * 0.1);
          setbacks.push({ fromTop, insetLeft: inset, insetRight: inset + (Math.random() - 0.5) * w * 0.04 });
        }
      }

      // Roof details
      const bTop = groundY - h;
      if (h > H * 0.45 && Math.random() < 0.35) {
        // Spire/antenna tower
        roofDetails.push({ type: "spire", xOff: w * 0.5, params: { height: 20 + Math.random() * 40, tipW: 1 + Math.random() * 2 } });
      }
      if (h > H * 0.35 && Math.random() < 0.3) {
        roofDetails.push({ type: "art-deco", xOff: 0, params: { tiers: 2 + Math.floor(Math.random() * 3) } });
      } else if (h > H * 0.3 && Math.random() < 0.25) {
        roofDetails.push({ type: "angular-crown", xOff: 0, params: { angle: Math.random() < 0.5 ? 0 : 1 } });
      } else if (h > H * 0.25 && Math.random() < 0.2) {
        roofDetails.push({ type: "stepped-crown", xOff: 0, params: { steps: 2 + Math.floor(Math.random() * 3) } });
      }
      if (Math.random() < 0.3) {
        roofDetails.push({ type: "water-tower", xOff: w * (0.2 + Math.random() * 0.5), params: { size: 3 + Math.random() * 4 } });
      }
      if (Math.random() < 0.25) {
        roofDetails.push({ type: "mech-penthouse", xOff: w * (0.1 + Math.random() * 0.3), params: { pw: w * (0.2 + Math.random() * 0.2), ph: 5 + Math.random() * 8 } });
      }
      if (h > H * 0.5 && Math.random() < 0.15) {
        roofDetails.push({ type: "helipad", xOff: w * 0.5, params: { radius: Math.min(w * 0.3, 12) } });
      }

      const antennaHeight = Math.random() < 0.2 ? 15 + Math.random() * 35 : 0;

      // Windows — uniform grid pattern (realistic)
      const windows: Window[] = [];
      const floorH = layer === 0 ? 7 : layer === 1 ? 9 : 11;
      const winW = layer === 0 ? 3 : layer === 1 ? 4 : 5;
      const winH = floorH * 0.55;
      const winGapX = layer === 0 ? 1.5 : layer === 1 ? 2 : 2.5;
      const winGapY = floorH - winH;
      const margin = 3;

      const cols = Math.floor((w - margin * 2) / (winW + winGapX));
      const rows = Math.floor((h - margin * 2) / floorH);

      // Determine lit pattern — some floors are mostly dark, some mostly lit (realistic office building at night)
      const floorLitBias: number[] = [];
      for (let r = 0; r < rows; r++) {
        floorLitBias.push(Math.random()); // per-floor bias
      }

      for (let r = 0; r < rows; r++) {
        // Apply setback narrowing
        let leftInset = 0, rightInset = 0;
        for (const sb of setbacks) {
          if (r * floorH < sb.fromTop) {
            leftInset = Math.max(leftInset, sb.insetLeft);
            rightInset = Math.max(rightInset, sb.insetRight);
          }
        }
        const rowStartX = x + margin + leftInset;
        const rowEndX = x + w - margin - rightInset;
        const rowCols = Math.floor((rowEndX - rowStartX) / (winW + winGapX));

        for (let c = 0; c < rowCols; c++) {
          const wx = rowStartX + c * (winW + winGapX);
          const wy = bTop + margin + r * floorH + winGapY * 0.5;

          // Realistic nighttime office building: ~30% lit overall, clustered by floor
          const floorBias = floorLitBias[r];
          const litChance = floorBias < 0.3 ? 0.05 : floorBias < 0.6 ? 0.25 : floorBias < 0.85 ? 0.45 : 0.7;
          const lit = Math.random() < litChance;
          // Warmth: 0 = cool white, 1 = warm yellow
          const warmth = Math.random() < 0.7 ? 0.6 + Math.random() * 0.4 : Math.random() * 0.3;

          windows.push({ wx, wy, ww: winW, wh: winH, lit, warmth, flicker: Math.random() * 6000 });
        }
      }

      // Glass reflection streak position
      const reflectionX = x + Math.random() * w * 0.3;
      const reflectionW = w * (0.15 + Math.random() * 0.25);

      return { x, w, h, layer, setbacks, roofDetails, antennaHeight, windows, reflectionX, reflectionW };
    }

    function generateCity() {
      layers = [[], [], []];
      streams = [];
      const groundY = H * 0.92;

      // Layer 0: far background — smaller, dimmer buildings
      let x = -10;
      while (x < W + 30) {
        const w = 15 + Math.random() * 35;
        const h = H * 0.12 + Math.random() * H * 0.32;
        layers[0].push(makeBuilding(x, w, h, 0, groundY));
        x += w + Math.random() * 4;
      }

      // Layer 1: midground
      x = -8;
      while (x < W + 25) {
        const roll = Math.random();
        let w: number, h: number;
        if (roll < 0.08) { w = 35 + Math.random() * 25; h = H * 0.55 + Math.random() * H * 0.2; }
        else if (roll < 0.25) { w = 25 + Math.random() * 40; h = H * 0.35 + Math.random() * H * 0.22; }
        else if (roll < 0.55) { w = 20 + Math.random() * 45; h = H * 0.2 + Math.random() * H * 0.2; }
        else { w = 18 + Math.random() * 35; h = H * 0.1 + Math.random() * H * 0.15; }
        layers[1].push(makeBuilding(x, w, h, 1, groundY));
        x += w + Math.random() * 3;
      }

      // Layer 2: foreground — largest, most detailed
      x = -5;
      while (x < W + 20) {
        const roll = Math.random();
        let w: number, h: number;
        if (roll < 0.05) { w = 45 + Math.random() * 30; h = H * 0.65 + Math.random() * H * 0.2; }
        else if (roll < 0.15) { w = 35 + Math.random() * 40; h = H * 0.5 + Math.random() * H * 0.2; }
        else if (roll < 0.35) { w = 30 + Math.random() * 50; h = H * 0.3 + Math.random() * H * 0.22; }
        else if (roll < 0.65) { w = 25 + Math.random() * 55; h = H * 0.18 + Math.random() * H * 0.18; }
        else { w = 20 + Math.random() * 40; h = H * 0.08 + Math.random() * H * 0.14; }
        layers[2].push(makeBuilding(x, w, h, 2, groundY));

        const bTop = groundY - h;
        const streamCount = Math.max(1, Math.floor(w / 22));
        for (let s = 0; s < streamCount; s++) {
          streams.push({
            x: x + 4 + Math.random() * (w - 8),
            y: bTop - Math.random() * 400,
            speed: 1.2 + Math.random() * 3.5,
            length: 6 + Math.floor(Math.random() * 14),
            chars: Array.from({ length: 25 }, () => CHARS[Math.floor(Math.random() * CHARS.length)]),
            opacity: 0.12 + Math.random() * 0.28,
            buildingTop: bTop,
          });
        }
        x += w + Math.random() * 3;
      }
    }

    generateCity();
    let time = 0;

    // ── Render a single building ──
    function drawBuilding(c: CanvasRenderingContext2D, b: Building, groundY: number, layerAlpha: number) {
      const bTop = groundY - b.h;

      // Build the building silhouette path with setbacks
      c.save();
      c.beginPath();
      // Start bottom-left, go up
      let curX = b.x;
      let curY = groundY;
      c.moveTo(curX, curY);

      // Left side going up with setbacks
      const sortedSetbacks = [...b.setbacks].sort((a, bb) => bb.fromTop - a.fromTop);
      let prevLeft = b.x;
      for (const sb of sortedSetbacks) {
        const sbY = bTop + sb.fromTop;
        c.lineTo(prevLeft, sbY);
        c.lineTo(b.x + sb.insetLeft, sbY);
        prevLeft = b.x + sb.insetLeft;
      }
      c.lineTo(prevLeft, bTop);
      // Top
      let prevRight = b.x + b.w;
      for (const sb of sortedSetbacks) {
        prevRight = b.x + b.w - sb.insetRight;
      }
      c.lineTo(prevRight, bTop);
      // Right side going down with setbacks
      const reversedSetbacks = [...b.setbacks].sort((a, bb) => a.fromTop - bb.fromTop);
      for (const sb of reversedSetbacks) {
        const sbY = bTop + sb.fromTop;
        c.lineTo(b.x + b.w - sb.insetRight, sbY);
        c.lineTo(b.x + b.w, sbY);
      }
      c.lineTo(b.x + b.w, groundY);
      c.closePath();

      // Main building fill — dark glass facade
      const bGrad = c.createLinearGradient(b.x, bTop, b.x + b.w, groundY);
      bGrad.addColorStop(0, `rgba(10, 20, 38, ${0.88 * layerAlpha})`);
      bGrad.addColorStop(0.3, `rgba(8, 16, 32, ${0.82 * layerAlpha})`);
      bGrad.addColorStop(0.7, `rgba(5, 12, 25, ${0.85 * layerAlpha})`);
      bGrad.addColorStop(1, `rgba(4, 9, 18, ${0.92 * layerAlpha})`);
      c.fillStyle = bGrad;
      c.fill();

      // Glass reflection streak — vertical lighter band
      const reflGrad = c.createLinearGradient(b.reflectionX, bTop, b.reflectionX + b.reflectionW, bTop);
      reflGrad.addColorStop(0, "transparent");
      reflGrad.addColorStop(0.3, `rgba(80, 140, 180, ${0.04 * layerAlpha})`);
      reflGrad.addColorStop(0.5, `rgba(100, 170, 210, ${0.06 * layerAlpha})`);
      reflGrad.addColorStop(0.7, `rgba(80, 140, 180, ${0.04 * layerAlpha})`);
      reflGrad.addColorStop(1, "transparent");
      c.fillStyle = reflGrad;
      c.fill();

      // Horizontal floor bands
      const floorH = b.layer === 0 ? 7 : b.layer === 1 ? 9 : 11;
      c.strokeStyle = `rgba(60, 90, 120, ${0.08 * layerAlpha})`;
      c.lineWidth = 0.4;
      for (let fy = bTop + floorH; fy < groundY; fy += floorH) {
        c.beginPath();
        c.moveTo(b.x, fy);
        c.lineTo(b.x + b.w, fy);
        c.stroke();
      }

      // Vertical mullion lines (structural)
      const panelW = Math.max(8, b.w / Math.max(1, Math.floor(b.w / 12)));
      c.strokeStyle = `rgba(50, 80, 110, ${0.05 * layerAlpha})`;
      c.lineWidth = 0.3;
      for (let fx = b.x + panelW; fx < b.x + b.w; fx += panelW) {
        c.beginPath();
        c.moveTo(fx, bTop);
        c.lineTo(fx, groundY);
        c.stroke();
      }

      // 3D depth — right face shadow
      const edgeW = Math.min(14, b.w * 0.12);
      const edgeGrad = c.createLinearGradient(b.x + b.w - edgeW, bTop, b.x + b.w, bTop);
      edgeGrad.addColorStop(0, "transparent");
      edgeGrad.addColorStop(1, `rgba(0, 0, 0, ${0.45 * layerAlpha})`);
      c.fillStyle = edgeGrad;
      c.fillRect(b.x + b.w - edgeW, bTop, edgeW, b.h);

      // Left edge highlight
      c.fillStyle = `rgba(100, 160, 200, ${0.06 * layerAlpha})`;
      c.fillRect(b.x, bTop, 1.2, b.h);

      // Roof line glow
      c.strokeStyle = `rgba(0, 229, 255, ${0.14 * layerAlpha})`;
      c.lineWidth = 1.2;
      c.beginPath();
      c.moveTo(b.x, bTop);
      c.lineTo(b.x + b.w, bTop);
      c.stroke();

      // Building outline
      c.strokeStyle = `rgba(40, 70, 100, ${0.1 * layerAlpha})`;
      c.lineWidth = 0.6;
      c.strokeRect(b.x, bTop, b.w, b.h);

      // ── Roof details ──
      for (const rd of b.roofDetails) {
        if (rd.type === "spire") {
          const sx = b.x + rd.xOff;
          const spH = rd.params.height;
          const tipW = rd.params.tipW;
          // Spire shaft
          c.strokeStyle = `rgba(80, 110, 140, ${0.3 * layerAlpha})`;
          c.lineWidth = tipW;
          c.beginPath();
          c.moveTo(sx, bTop);
          c.lineTo(sx, bTop - spH);
          c.stroke();
          // Spire tip light
          const blink = Math.sin(time * 0.04 + b.x * 0.1);
          if (blink > 0) {
            c.fillStyle = `rgba(255, 40, 40, ${blink * 0.85 * layerAlpha})`;
            c.beginPath();
            c.arc(sx, bTop - spH, 2, 0, Math.PI * 2);
            c.fill();
            c.fillStyle = `rgba(255, 40, 40, ${blink * 0.2 * layerAlpha})`;
            c.beginPath();
            c.arc(sx, bTop - spH, 7, 0, Math.PI * 2);
            c.fill();
          }
        } else if (rd.type === "art-deco") {
          // Tiered art-deco crown
          const tiers = rd.params.tiers;
          for (let t = 0; t < tiers; t++) {
            const tierW = b.w * (0.8 - t * 0.15);
            const tierH = 6 + t * 3;
            const tx = b.x + (b.w - tierW) / 2;
            const ty = bTop - (t + 1) * tierH;
            c.fillStyle = `rgba(12, 24, 42, ${0.9 * layerAlpha})`;
            c.fillRect(tx, ty, tierW, tierH);
            c.strokeStyle = `rgba(0, 229, 255, ${0.1 * layerAlpha})`;
            c.lineWidth = 0.5;
            c.strokeRect(tx, ty, tierW, tierH);
          }
        } else if (rd.type === "angular-crown") {
          // Angled/sloped top
          const crownH = 15 + Math.random() * 10;
          c.fillStyle = `rgba(10, 22, 40, ${0.9 * layerAlpha})`;
          c.beginPath();
          if (rd.params.angle === 0) {
            // Peaked
            c.moveTo(b.x, bTop);
            c.lineTo(b.x + b.w * 0.5, bTop - crownH);
            c.lineTo(b.x + b.w, bTop);
          } else {
            // Slanted
            c.moveTo(b.x, bTop);
            c.lineTo(b.x + b.w * 0.3, bTop - crownH);
            c.lineTo(b.x + b.w, bTop - crownH * 0.4);
            c.lineTo(b.x + b.w, bTop);
          }
          c.closePath();
          c.fill();
          c.strokeStyle = `rgba(0, 229, 255, ${0.1 * layerAlpha})`;
          c.lineWidth = 0.5;
          c.stroke();
        } else if (rd.type === "stepped-crown") {
          const steps = rd.params.steps;
          const stepH = 5;
          for (let s = 0; s < steps; s++) {
            const sw = b.w * (1 - (s + 1) * 0.2);
            const sx = b.x + (b.w - sw) / 2;
            const sy = bTop - (s + 1) * stepH;
            c.fillStyle = `rgba(10, 20, 38, ${0.9 * layerAlpha})`;
            c.fillRect(sx, sy, sw, stepH);
            c.strokeStyle = `rgba(50, 80, 110, ${0.08 * layerAlpha})`;
            c.lineWidth = 0.4;
            c.strokeRect(sx, sy, sw, stepH);
          }
        } else if (rd.type === "water-tower") {
          const wtX = b.x + rd.xOff;
          const sz = rd.params.size;
          // Legs
          c.strokeStyle = `rgba(70, 90, 110, ${0.2 * layerAlpha})`;
          c.lineWidth = 0.8;
          c.beginPath();
          c.moveTo(wtX - sz * 0.5, bTop);
          c.lineTo(wtX - sz * 0.3, bTop - sz * 2);
          c.moveTo(wtX + sz * 0.5, bTop);
          c.lineTo(wtX + sz * 0.3, bTop - sz * 2);
          c.stroke();
          // Tank
          c.fillStyle = `rgba(15, 28, 45, ${0.8 * layerAlpha})`;
          c.fillRect(wtX - sz * 0.5, bTop - sz * 3, sz, sz);
          c.strokeStyle = `rgba(50, 80, 110, ${0.15 * layerAlpha})`;
          c.lineWidth = 0.4;
          c.strokeRect(wtX - sz * 0.5, bTop - sz * 3, sz, sz);
        } else if (rd.type === "mech-penthouse") {
          const mpX = b.x + rd.xOff;
          const pw = rd.params.pw;
          const ph = rd.params.ph;
          c.fillStyle = `rgba(12, 22, 38, ${0.85 * layerAlpha})`;
          c.fillRect(mpX, bTop - ph, pw, ph);
          c.strokeStyle = `rgba(50, 80, 110, ${0.1 * layerAlpha})`;
          c.lineWidth = 0.4;
          c.strokeRect(mpX, bTop - ph, pw, ph);
        } else if (rd.type === "helipad") {
          const hx = b.x + rd.xOff;
          const hr = rd.params.radius;
          c.strokeStyle = `rgba(0, 229, 255, ${0.12 * layerAlpha})`;
          c.lineWidth = 0.8;
          c.beginPath();
          c.arc(hx, bTop - 2, hr, 0, Math.PI * 2);
          c.stroke();
          // H marking
          c.strokeStyle = `rgba(0, 229, 255, ${0.08 * layerAlpha})`;
          c.lineWidth = 1;
          c.beginPath();
          c.moveTo(hx - hr * 0.3, bTop - 2 - hr * 0.35);
          c.lineTo(hx - hr * 0.3, bTop - 2 + hr * 0.35);
          c.moveTo(hx + hr * 0.3, bTop - 2 - hr * 0.35);
          c.lineTo(hx + hr * 0.3, bTop - 2 + hr * 0.35);
          c.moveTo(hx - hr * 0.3, bTop - 2);
          c.lineTo(hx + hr * 0.3, bTop - 2);
          c.stroke();
        }
      }

      // Antenna with blinking red light
      if (b.antennaHeight > 0) {
        const ax = b.x + b.w * 0.5;
        c.strokeStyle = `rgba(80, 100, 130, ${0.25 * layerAlpha})`;
        c.lineWidth = 0.8;
        c.beginPath();
        c.moveTo(ax, bTop);
        c.lineTo(ax, bTop - b.antennaHeight);
        c.stroke();
        const blinkPhase = Math.sin(time * 0.04 + b.x * 0.1);
        if (blinkPhase > 0) {
          const alpha = blinkPhase * 0.85 * layerAlpha;
          c.fillStyle = `rgba(255, 40, 40, ${alpha})`;
          c.beginPath();
          c.arc(ax, bTop - b.antennaHeight, 2, 0, Math.PI * 2);
          c.fill();
          c.fillStyle = `rgba(255, 40, 40, ${alpha * 0.25})`;
          c.beginPath();
          c.arc(ax, bTop - b.antennaHeight, 7, 0, Math.PI * 2);
          c.fill();
        }
      }

      // ── Windows — realistic nighttime office ──
      for (const win of b.windows) {
        if (!win.lit) {
          // Dark window — faint glass reflection
          c.fillStyle = `rgba(20, 35, 55, ${0.15 * layerAlpha})`;
          c.fillRect(win.wx, win.wy, win.ww, win.wh);
          // Tiny chance to flicker on
          if (Math.random() < 0.0003) win.lit = true;
          continue;
        }
        if (Math.random() < 0.0003) { win.lit = false; continue; }

        const flick = Math.sin(time * 0.008 + win.flicker) * 0.5 + 0.5;
        const baseAlpha = (0.15 + flick * 0.35) * layerAlpha;

        // Window color based on warmth (warm yellow to cool white)
        const r = Math.floor(200 + win.warmth * 55);
        const g = Math.floor(170 + win.warmth * 50);
        const bVal = Math.floor(80 + (1 - win.warmth) * 120);

        // Inner glow
        c.fillStyle = `rgba(${r}, ${g}, ${bVal}, ${baseAlpha})`;
        c.fillRect(win.wx, win.wy, win.ww, win.wh);

        // Outer bleed glow
        c.fillStyle = `rgba(${r}, ${g}, ${bVal}, ${baseAlpha * 0.12})`;
        c.fillRect(win.wx - 0.8, win.wy - 0.5, win.ww + 1.6, win.wh + 1);
      }

      c.restore();
    }

    function draw() {
      if (document.hidden) { animId = requestAnimationFrame(draw); return; }
      time++;
      const c = ctx!;
      c.clearRect(0, 0, W, H);

      // Sky gradient
      const grad = c.createLinearGradient(0, 0, 0, H);
      grad.addColorStop(0, "#010306");
      grad.addColorStop(0.3, "#020610");
      grad.addColorStop(0.6, "#040a1a");
      grad.addColorStop(1, "#050c1e");
      c.fillStyle = grad;
      c.fillRect(0, 0, W, H);

      const groundY = H * 0.92;

      // ── Layer 0 (background) — faint, atmospheric ──
      for (const b of layers[0]) drawBuilding(c, b, groundY, 0.35);

      // Fog between back and mid layer
      const fog0 = c.createLinearGradient(0, groundY - H * 0.45, 0, groundY);
      fog0.addColorStop(0, "transparent");
      fog0.addColorStop(0.4, "rgba(8, 18, 35, 0.3)");
      fog0.addColorStop(0.8, "rgba(6, 14, 28, 0.2)");
      fog0.addColorStop(1, "transparent");
      c.fillStyle = fog0;
      c.fillRect(0, groundY - H * 0.45, W, H * 0.45);

      // ── Layer 1 (midground) ──
      for (const b of layers[1]) drawBuilding(c, b, groundY, 0.6);

      // Fog between mid and foreground
      const fog1 = c.createLinearGradient(0, groundY - H * 0.3, 0, groundY);
      fog1.addColorStop(0, "transparent");
      fog1.addColorStop(0.5, "rgba(6, 15, 30, 0.2)");
      fog1.addColorStop(1, "transparent");
      c.fillStyle = fog1;
      c.fillRect(0, groundY - H * 0.3, W, H * 0.3);

      // ── Layer 2 (foreground) — full detail ──
      for (const b of layers[2]) drawBuilding(c, b, groundY, 1.0);

      // ── Ambient glow at skyline base ──
      const ambientGlow = c.createRadialGradient(W * 0.5, groundY, 0, W * 0.5, groundY, W * 0.6);
      ambientGlow.addColorStop(0, "rgba(0, 180, 220, 0.025)");
      ambientGlow.addColorStop(0.5, "rgba(0, 140, 180, 0.012)");
      ambientGlow.addColorStop(1, "transparent");
      c.fillStyle = ambientGlow;
      c.fillRect(0, groundY - H * 0.4, W, H * 0.42);

      // Ground line
      c.strokeStyle = "rgba(0, 229, 255, 0.08)";
      c.lineWidth = 1;
      c.beginPath();
      c.moveTo(0, groundY);
      c.lineTo(W, groundY);
      c.stroke();

      // Ground fog
      const fogGrad = c.createLinearGradient(0, groundY - 30, 0, groundY + 10);
      fogGrad.addColorStop(0, "transparent");
      fogGrad.addColorStop(0.5, "rgba(0, 229, 255, 0.015)");
      fogGrad.addColorStop(1, "rgba(0, 229, 255, 0.008)");
      c.fillStyle = fogGrad;
      c.fillRect(0, groundY - 30, W, 40);

      // ── Binary streams (unchanged) ──
      c.font = "9px 'JetBrains Mono', 'Courier New', monospace";
      for (const s of streams) {
        s.y += s.speed;
        if (s.y > s.buildingTop + 15) {
          s.y = s.buildingTop - 100 - Math.random() * 350;
          for (let i = 0; i < s.chars.length; i++) {
            if (Math.random() < 0.4) s.chars[i] = CHARS[Math.floor(Math.random() * 2)];
          }
        }
        for (let i = 0; i < s.length; i++) {
          const cy = s.y + i * 10;
          if (cy < 0 || cy > s.buildingTop + 8) continue;
          const fade = 1 - (i / s.length);
          const alpha = s.opacity * fade;
          c.fillStyle = i === 0
            ? `rgba(180, 255, 255, ${Math.min(alpha * 1.8, 0.6)})`
            : `rgba(0, 229, 255, ${alpha})`;
          c.fillText(s.chars[i % s.chars.length], s.x, cy);
          if (Math.random() < 0.04) s.chars[i % s.chars.length] = CHARS[Math.floor(Math.random() * 2)];
        }
      }

      // Scan line
      const scanY = (time * 1.2) % (H * 2) - H * 0.3;
      const scanGrad = c.createLinearGradient(0, scanY - 15, 0, scanY + 15);
      scanGrad.addColorStop(0, "transparent");
      scanGrad.addColorStop(0.5, "rgba(0, 229, 255, 0.025)");
      scanGrad.addColorStop(1, "transparent");
      c.fillStyle = scanGrad;
      c.fillRect(0, scanY - 15, W, 30);

      animId = requestAnimationFrame(draw);
    }

    draw();
    return () => {
      cancelAnimationFrame(animId);
      window.removeEventListener("resize", handleResize);
    };
  }, []);

  return <canvas ref={canvasRef} style={{ position: "absolute", inset: 0, width: "100%", height: "100%" }} />;
}

// ── Login Page ───────────────────────────────────────────────────────────────

export default function LoginPage({ onLogin }: LoginProps) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [confirmNewPassword, setConfirmNewPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [resetSent, setResetSent] = useState(false);
  const [signupSuccess, setSignupSuccess] = useState(false);
  const [resetComplete, setResetComplete] = useState(false);

  // Check if URL has a reset token (Bug 18: move to state)
  const urlParams = new URLSearchParams(window.location.search);
  const [resetToken, setResetToken] = useState<string | null>(urlParams.get("token"));
  const [mode, setMode] = useState<"login" | "reset" | "signup" | "reset-confirm">(resetToken ? "reset-confirm" : "login");

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      const res = await fetch("/api/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password }),
        credentials: "include",
      });
      const data = await res.json();
      if (data.ok) {
        queryClient.invalidateQueries({ queryKey: ["/api/auth/me"] });
        onLogin?.();
      } else {
        setError(data.error || "Invalid credentials");
      }
    } catch {
      setError("Connection failed");
    }
    setLoading(false);
  };

  const handleReset = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      const res = await fetch("/api/auth/reset", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email }),
        credentials: "include",
      });
      const data = await res.json();
      if (data.ok !== false) {
        setResetSent(true);
      } else {
        setError(data.error || "Reset failed");
      }
    } catch {
      setError("Reset failed");
    }
    setLoading(false);
  };

  const handleSignup = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    if (password !== confirmPassword) {
      setError("Passwords do not match");
      return;
    }
    if (password.length < 6) {
      setError("Password must be at least 6 characters");
      return;
    }
    setLoading(true);
    try {
      const res = await fetch("/api/auth/register", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password }),
        credentials: "include",
      });
      const data = await res.json();
      if (data.ok) {
        // Auto-login after registration
        const loginRes = await fetch("/api/auth/login", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ email, password }),
          credentials: "include",
        });
        const loginData = await loginRes.json();
        if (loginData.ok) {
          queryClient.invalidateQueries({ queryKey: ["/api/auth/me"] });
          onLogin?.();
        } else {
          setSignupSuccess(true);
          setMode("login");
        }
      } else {
        setError(data.error || "Registration failed");
      }
    } catch {
      setError("Connection failed");
    }
    setLoading(false);
  };

  const handleResetConfirm = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    if (newPassword !== confirmNewPassword) {
      setError("Passwords do not match");
      return;
    }
    if (newPassword.length < 6) {
      setError("Password must be at least 6 characters");
      return;
    }
    setLoading(true);
    try {
      const res = await fetch("/api/auth/reset-confirm", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ token: resetToken, newPassword }),
        credentials: "include",
      });
      const data = await res.json();
      if (data.ok) {
        setResetComplete(true);
        // Clear the token from URL
        window.history.replaceState({}, "", window.location.pathname);
        setResetToken(null);
      } else {
        setError(data.error || "Reset failed — link may have expired");
      }
    } catch {
      setError("Connection failed");
    }
    setLoading(false);
  };

  const inputStyle: React.CSSProperties = {
    width: "100%", padding: "12px 14px", marginBottom: 12,
    background: "rgba(0, 15, 30, 0.7)",
    border: "1px solid rgba(0, 229, 255, 0.15)",
    borderRadius: 4, color: "#c8d6e5", fontSize: 13,
    fontFamily: "'JetBrains Mono', monospace",
    outline: "none", boxSizing: "border-box",
    letterSpacing: "0.03em",
  };

  return (
    <div style={{
      minHeight: "100dvh",
      background: "#020408",
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      position: "relative",
      overflow: "hidden",
    }}>
      {/* Animated Matrix city */}
      <CityMatrixCanvas />

      {/* Binary data streams overlay */}
      <div style={{ position: "absolute", inset: 0, overflow: "hidden", zIndex: 1, opacity: 0.08, pointerEvents: "none" }}>
        {Array.from({length: 15}, (_, i) => (
          <div key={i} style={{
            position: "absolute",
            left: `${3 + i * 6.5}%`,
            top: "-100%",
            width: "1px",
            height: "200%",
            background: "linear-gradient(180deg, transparent 0%, #00e5ff 30%, #00e5ff 50%, transparent 100%)",
            animation: `matrixFall ${3 + (i % 4) * 2}s linear infinite`,
            animationDelay: `${(i * 0.7) % 5}s`,
          }} />
        ))}
      </div>

      {/* Scan line */}
      <div style={{
        position: "absolute", inset: 0, zIndex: 2, pointerEvents: "none", overflow: "hidden",
      }}>
        <div style={{
          position: "absolute", left: 0, right: 0, height: "1px",
          background: "rgba(0, 229, 255, 0.06)",
          animation: "scanLine 6s linear infinite",
        }} />
      </div>

      {/* Corner HUD markers */}
      <div className="hud-label-tl" style={{ position: "absolute", top: 16, left: 16, zIndex: 20, fontSize: 12, color: "#00e5ff", opacity: 0.5, fontFamily: "'JetBrains Mono', monospace", letterSpacing: "0.15em", fontWeight: 600 }}>
        VOLTRADEAI
      </div>
      <div className="hud-status-bl" style={{ position: "absolute", bottom: 16, left: 16, zIndex: 20, fontSize: 9, color: "#3a4a5c", fontFamily: "'JetBrains Mono', monospace", letterSpacing: "0.1em" }}>
        SYS: ONLINE | ML: 36 FEATURES | ALPACA: CONNECTED
      </div>
      <div className="hud-status-br" style={{ position: "absolute", bottom: 16, right: 16, zIndex: 20, fontSize: 9, color: "#3a4a5c", fontFamily: "'JetBrains Mono', monospace" }}>
        v2.0 // PAPER TRADING MODE
      </div>

      <style>{`
        @keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0.2; } }
        @keyframes matrixFall { 0% { transform: translateY(0); } 100% { transform: translateY(100vh); } }
        @keyframes scanLine { 0% { top: -2px; } 100% { top: 100%; } }

        @media (max-width: 600px) {
          .login-card {
            padding: 1.5rem 1.25rem !important;
            max-width: 320px !important;
            margin: 0 0.75rem !important;
          }
          .hud-label-tl {
            left: auto !important;
            right: 16px !important;
            top: 16px !important;
          }
          .hud-status-bl, .hud-status-br {
            left: 50% !important;
            right: auto !important;
            transform: translateX(-50%);
            text-align: center;
            white-space: nowrap;
          }
          .hud-status-bl {
            bottom: 28px !important;
          }
          .hud-status-br {
            bottom: 12px !important;
          }
        }
      `}</style>

      {/* Login card */}
      <div className="login-card" style={{
        position: "relative",
        zIndex: 10,
        width: "100%",
        maxWidth: 380,
        padding: "2.5rem 2rem",
        background: "rgba(3, 5, 10, 0.92)",
        border: "1px solid rgba(0, 229, 255, 0.12)",
        borderRadius: 6,
        boxShadow: "0 0 60px rgba(0, 0, 0, 0.8), 0 0 30px rgba(0, 229, 255, 0.03)",
        backdropFilter: "blur(12px)",
        margin: "0 1rem",
      }}>
        <div style={{ textAlign: "center", marginBottom: "0.5rem" }}>
          <span style={{ fontSize: 9, letterSpacing: "0.2em", color: "#00e5ff", opacity: 0.4, fontFamily: "'JetBrains Mono', monospace" }}>
            TOP SECRET // AUTHORIZED PERSONNEL ONLY
          </span>
        </div>

        <div style={{ textAlign: "center", marginBottom: "1.5rem" }}>
          <div style={{ fontSize: 28, fontWeight: 800, letterSpacing: "0.08em", fontFamily: "'Inter', 'JetBrains Mono', sans-serif" }}>
            <span style={{ color: "#ffffff" }}>VolTrade</span>
            <span style={{ color: "#00e5ff" }}>AI</span>
          </div>
          <div style={{ fontSize: 11, color: "#4a5c70", letterSpacing: "0.15em", marginTop: 6, textTransform: "uppercase", fontFamily: "'JetBrains Mono', monospace" }}>
            {mode === "login" ? "SIGN IN TO YOUR ACCOUNT" : mode === "signup" ? "CREATE AN ACCOUNT" : mode === "reset-confirm" ? "SET NEW PASSWORD" : "RESET YOUR PASSWORD"}
          </div>
        </div>

        {signupSuccess && mode === "login" && (
          <div style={{ color: "#30d158", fontSize: 12, marginBottom: 12, textAlign: "center", fontFamily: "'JetBrains Mono', monospace" }}>
            Account created! Sign in below.
          </div>
        )}

        {mode === "login" ? (
          <form onSubmit={handleSubmit}>
            <input
              type="email" placeholder="Email" value={email}
              onChange={e => setEmail(e.target.value)} required
              style={inputStyle}
              onFocus={e => { e.target.style.borderColor = "rgba(0, 229, 255, 0.4)"; e.target.style.boxShadow = "0 0 12px rgba(0, 229, 255, 0.08)"; }}
              onBlur={e => { e.target.style.borderColor = "rgba(0, 229, 255, 0.15)"; e.target.style.boxShadow = "none"; }}
            />
            <input
              type="password" placeholder="Password" value={password}
              onChange={e => setPassword(e.target.value)} required
              style={{ ...inputStyle, marginBottom: 16 }}
              onFocus={e => { e.target.style.borderColor = "rgba(0, 229, 255, 0.4)"; e.target.style.boxShadow = "0 0 12px rgba(0, 229, 255, 0.08)"; }}
              onBlur={e => { e.target.style.borderColor = "rgba(0, 229, 255, 0.15)"; e.target.style.boxShadow = "none"; }}
            />
            {error && <div style={{ color: "#ff3333", fontSize: 12, marginBottom: 12, textAlign: "center", fontFamily: "'JetBrains Mono', monospace" }}>{error}</div>}
            <button type="submit" disabled={loading} style={{
              width: "100%", padding: "12px",
              background: "rgba(0, 229, 255, 0.08)",
              border: "1px solid rgba(0, 229, 255, 0.3)",
              borderRadius: 4, color: "#00e5ff", fontSize: 12,
              fontWeight: 600, letterSpacing: "0.15em", textTransform: "uppercase",
              fontFamily: "'JetBrains Mono', monospace",
              cursor: "pointer", transition: "all 200ms ease",
            }}
              onMouseOver={e => { (e.target as HTMLElement).style.background = "rgba(0, 229, 255, 0.15)"; (e.target as HTMLElement).style.boxShadow = "0 0 20px rgba(0, 229, 255, 0.1)"; }}
              onMouseOut={e => { (e.target as HTMLElement).style.background = "rgba(0, 229, 255, 0.08)"; (e.target as HTMLElement).style.boxShadow = "none"; }}
            >
              {loading ? "AUTHENTICATING..." : "SIGN IN"}
            </button>
          </form>
        ) : mode === "signup" ? (
          <form onSubmit={handleSignup}>
            <input
              type="email" placeholder="Email" value={email}
              onChange={e => setEmail(e.target.value)} required
              style={inputStyle}
              onFocus={e => { e.target.style.borderColor = "rgba(0, 229, 255, 0.4)"; e.target.style.boxShadow = "0 0 12px rgba(0, 229, 255, 0.08)"; }}
              onBlur={e => { e.target.style.borderColor = "rgba(0, 229, 255, 0.15)"; e.target.style.boxShadow = "none"; }}
            />
            <input
              type="password" placeholder="Password (min 6 chars)" value={password}
              onChange={e => setPassword(e.target.value)} required
              style={inputStyle}
              onFocus={e => { e.target.style.borderColor = "rgba(0, 229, 255, 0.4)"; e.target.style.boxShadow = "0 0 12px rgba(0, 229, 255, 0.08)"; }}
              onBlur={e => { e.target.style.borderColor = "rgba(0, 229, 255, 0.15)"; e.target.style.boxShadow = "none"; }}
            />
            <input
              type="password" placeholder="Confirm Password" value={confirmPassword}
              onChange={e => setConfirmPassword(e.target.value)} required
              style={{ ...inputStyle, marginBottom: 16 }}
              onFocus={e => { e.target.style.borderColor = "rgba(0, 229, 255, 0.4)"; e.target.style.boxShadow = "0 0 12px rgba(0, 229, 255, 0.08)"; }}
              onBlur={e => { e.target.style.borderColor = "rgba(0, 229, 255, 0.15)"; e.target.style.boxShadow = "none"; }}
            />
            {error && <div style={{ color: "#ff3333", fontSize: 12, marginBottom: 12, textAlign: "center", fontFamily: "'JetBrains Mono', monospace" }}>{error}</div>}
            <button type="submit" disabled={loading} style={{
              width: "100%", padding: "12px",
              background: "rgba(0, 229, 255, 0.08)",
              border: "1px solid rgba(0, 229, 255, 0.3)",
              borderRadius: 4, color: "#00e5ff", fontSize: 12,
              fontWeight: 600, letterSpacing: "0.15em", textTransform: "uppercase",
              fontFamily: "'JetBrains Mono', monospace",
              cursor: "pointer", transition: "all 200ms ease",
            }}
              onMouseOver={e => { (e.target as HTMLElement).style.background = "rgba(0, 229, 255, 0.15)"; (e.target as HTMLElement).style.boxShadow = "0 0 20px rgba(0, 229, 255, 0.1)"; }}
              onMouseOut={e => { (e.target as HTMLElement).style.background = "rgba(0, 229, 255, 0.08)"; (e.target as HTMLElement).style.boxShadow = "none"; }}
            >
              {loading ? "CREATING ACCOUNT..." : "CREATE ACCOUNT"}
            </button>
          </form>
        ) : (
          mode === "reset-confirm" ? (
          <form onSubmit={handleResetConfirm}>
            {resetComplete ? (
              <div style={{ textAlign: "center", padding: "1rem 0" }}>
                <div style={{ color: "#30d158", fontSize: 13, marginBottom: 12, fontFamily: "'JetBrains Mono', monospace" }}>Password reset successfully</div>
                <button
                  type="button"
                  onClick={() => { setMode("login"); setResetComplete(false); setNewPassword(""); setConfirmNewPassword(""); }}
                  style={{
                    width: "100%", padding: "12px",
                    background: "rgba(0, 229, 255, 0.08)",
                    border: "1px solid rgba(0, 229, 255, 0.3)",
                    borderRadius: 4, color: "#00e5ff", fontSize: 12,
                    fontWeight: 600, letterSpacing: "0.15em", textTransform: "uppercase",
                    fontFamily: "'JetBrains Mono', monospace", cursor: "pointer",
                  }}
                >
                  SIGN IN WITH NEW PASSWORD
                </button>
              </div>
            ) : (
              <>
                <input
                  type="password" placeholder="New Password" value={newPassword}
                  onChange={e => setNewPassword(e.target.value)} required
                  style={inputStyle}
                  onFocus={e => { e.target.style.borderColor = "rgba(0, 229, 255, 0.4)"; e.target.style.boxShadow = "0 0 12px rgba(0, 229, 255, 0.08)"; }}
                  onBlur={e => { e.target.style.borderColor = "rgba(0, 229, 255, 0.15)"; e.target.style.boxShadow = "none"; }}
                />
                <input
                  type="password" placeholder="Confirm New Password" value={confirmNewPassword}
                  onChange={e => setConfirmNewPassword(e.target.value)} required
                  style={{ ...inputStyle, marginBottom: 16 }}
                  onFocus={e => { e.target.style.borderColor = "rgba(0, 229, 255, 0.4)"; e.target.style.boxShadow = "0 0 12px rgba(0, 229, 255, 0.08)"; }}
                  onBlur={e => { e.target.style.borderColor = "rgba(0, 229, 255, 0.15)"; e.target.style.boxShadow = "none"; }}
                />
                {error && <div style={{ color: "#ff453a", fontSize: 12, marginBottom: 12, textAlign: "center", fontFamily: "'JetBrains Mono', monospace" }}>{error}</div>}
                <button type="submit" disabled={loading} style={{
                  width: "100%", padding: "12px",
                  background: "rgba(0, 229, 255, 0.08)",
                  border: "1px solid rgba(0, 229, 255, 0.3)",
                  borderRadius: 4, color: "#00e5ff", fontSize: 12,
                  fontWeight: 600, letterSpacing: "0.15em", textTransform: "uppercase",
                  fontFamily: "'JetBrains Mono', monospace", cursor: "pointer",
                }}>
                  {loading ? "RESETTING..." : "RESET PASSWORD"}
                </button>
              </>
            )}
          </form>
        ) :
          <form onSubmit={handleReset}>
            {resetSent ? (
              <div style={{ textAlign: "center", color: "#00e5ff", fontSize: 13, padding: "1rem 0", fontFamily: "'JetBrains Mono', monospace" }}>
                Reset link sent to {email}
              </div>
            ) : (
              <>
                <input type="email" placeholder="Email" value={email}
                  onChange={e => setEmail(e.target.value)} required
                  style={{ ...inputStyle, marginBottom: 16 }}
                />
                {error && <div style={{ color: "#ff3333", fontSize: 12, marginBottom: 12, textAlign: "center" }}>{error}</div>}
                <button type="submit" disabled={loading} style={{
                  width: "100%", padding: "12px",
                  background: "rgba(0, 229, 255, 0.08)",
                  border: "1px solid rgba(0, 229, 255, 0.3)",
                  borderRadius: 4, color: "#00e5ff", fontSize: 12,
                  fontWeight: 600, letterSpacing: "0.15em", textTransform: "uppercase",
                  fontFamily: "'JetBrains Mono', monospace", cursor: "pointer",
                }}>
                  {loading ? "SENDING..." : "SEND RESET LINK"}
                </button>
              </>
            )}
          </form>
        )}

        <div style={{ textAlign: "center", marginTop: 16 }}>
          <button
            onClick={() => { setMode(mode === "login" ? "reset" : "login"); setError(""); setResetSent(false); setEmail(""); setPassword(""); }}
            style={{
              background: "none", border: "none", color: "#d4a017",
              fontSize: 11, cursor: "pointer", letterSpacing: "0.1em",
              fontFamily: "'JetBrains Mono', monospace", opacity: 0.7,
            }}
          >
            {mode === "login" ? "Forgot Password?" : "← Back to Sign In"}
          </button>
        </div>

        <div style={{ textAlign: "center", marginTop: 12 }}>
          <button
            onClick={() => { setMode(mode === "signup" ? "login" : "signup"); setError(""); setConfirmPassword(""); setEmail(""); setPassword(""); }}
            style={{
              background: "none", border: "none", color: "#00e5ff",
              fontSize: 11, cursor: "pointer", letterSpacing: "0.08em",
              fontFamily: "'JetBrains Mono', monospace", opacity: 0.6,
            }}
          >
            {mode === "signup" ? "Already have an account? Sign In" : "Don't have an account? Create one"}
          </button>
        </div>
      </div>
    </div>
  );
}
