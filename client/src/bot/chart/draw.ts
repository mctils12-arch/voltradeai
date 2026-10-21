// Bot page chart — the canvas renderer. A pure function of a Frame: no
// state, no DOM queries, no clock. BotChart.tsx owns the canvas, the one
// frame loop and the springs, and calls draw() once per dirty frame.
// Colours are the page's CSS tokens (DESIGN.md), read once by readTheme().
import {
  type Bar, type Viewport, type Timeframe, type PriceRange,
  xOf, pxPerBar, priceRange, priceTicks, priceDecimals, fmtPrice, yOf, priceOfY, timeTicks, visibleIndexRange, indexAtX, fmtBarTime,
} from "./scales";

export interface Theme {
  text: string; textDim: string; textFaint: string; grid: string; gridStrong: string;
  up: string; down: string; accent: string; red: string; green: string; orange: string; purple: string;
  surface: string; font: string;
}
export const FALLBACK_THEME: Theme = {
  text: "#eef3fb", textDim: "#b3c2d8", textFaint: "#6680a0", grid: "rgba(120, 165, 220, 0.08)", gridStrong: "rgba(120, 165, 220, 0.16)",
  up: "#4ade80", down: "#ff5a6e", accent: "#4d9fff", red: "#ff5a6e", green: "#4ade80", orange: "#fbb24c", purple: "#c084fc",
  surface: "#0a1628", font: "'Geist Mono', ui-monospace, 'JetBrains Mono', 'SF Mono', monospace",
};

/** The page's tokens, with DESIGN.md's canonical values as the fallback. */
export function readTheme(el: Element | null): Theme {
  if (!el || typeof getComputedStyle !== "function") return FALLBACK_THEME;
  const cs = getComputedStyle(el);
  const v = (name: string, fb: string) => { const x = cs.getPropertyValue(name).trim(); return x || fb; };
  return {
    text: v("--text-primary", FALLBACK_THEME.text), textDim: v("--text-secondary", FALLBACK_THEME.textDim), textFaint: v("--text-tertiary", FALLBACK_THEME.textFaint),
    grid: v("--border-subtle", FALLBACK_THEME.grid), gridStrong: v("--border", FALLBACK_THEME.gridStrong),
    up: v("--accent-green", FALLBACK_THEME.up), down: v("--accent-red", FALLBACK_THEME.down), accent: v("--accent", FALLBACK_THEME.accent),
    red: v("--accent-red", FALLBACK_THEME.red), green: v("--accent-green", FALLBACK_THEME.green), orange: v("--accent-orange", FALLBACK_THEME.orange), purple: v("--accent-purple", FALLBACK_THEME.purple),
    surface: v("--bg-secondary", FALLBACK_THEME.surface), font: v("--font-mono", FALLBACK_THEME.font),
  };
}

/** A colour with its alpha replaced: #rgb / #rrggbb / rgb() / rgba(). */
export function withAlpha(color: string, alpha: number): string {
  const a = Math.max(0, Math.min(1, alpha));
  const hex = /^#([0-9a-f]{3}|[0-9a-f]{6})$/i.exec(color.trim());
  if (hex) {
    let h = hex[1];
    if (h.length === 3) h = h.split("").map((c) => c + c).join("");
    const r = parseInt(h.slice(0, 2), 16), g = parseInt(h.slice(2, 4), 16), b = parseInt(h.slice(4, 6), 16);
    return `rgba(${r}, ${g}, ${b}, ${a})`;
  }
  const rgb = /^rgba?\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)/i.exec(color.trim());
  if (rgb) return `rgba(${rgb[1]}, ${rgb[2]}, ${rgb[3]}, ${a})`;
  return color;
}

export type Dash = "solid" | "dash" | "dot";
export interface DrawLevel {
  price: number; color: string; dash: Dash; label: string;
  /** a muted level is drawn faint — pending, done or suppressed */
  muted?: boolean;
  /** false = the level may sit outside the view (the hard floor); default true keeps it in the price range */
  fit?: boolean;
}
export interface DrawZone { lo: number; hi: number; color: string; alpha: number; label?: string }
export interface DrawMarker { index: number; price: number; color: string; label?: string; shape: "up" | "down" }
export interface Frame {
  bars: readonly Bar[]; vp: Viewport; w: number; h: number; tf: Timeframe; theme: Theme;
  levels: readonly DrawLevel[]; zones: readonly DrawZone[]; markers: readonly DrawMarker[];
  cross: { x: number; y: number } | null;
  showVolume: boolean;
}
export interface DrawResult { range: PriceRange; decimals: number; crossIndex: number | null; crossPrice: number | null }

export const GUTTER_R = 62;
export const AXIS_B = 18;
export const VOL_FRAC = 0.16;
export const LABEL_H = 14;

export interface Geometry { plotW: number; paneH: number; priceH: number; volTop: number; volH: number }
export function geometry(w: number, h: number, showVolume: boolean): Geometry {
  const plotW = Math.max(1, w - GUTTER_R);
  const paneH = Math.max(1, h - AXIS_B);
  const priceH = showVolume ? paneH * (1 - VOL_FRAC) : paneH;
  return { plotW, paneH, priceH, volTop: priceH, volH: paneH - priceH };
}

const DASHES: Record<Dash, number[]> = { solid: [], dash: [6, 4], dot: [2, 3] };

/** Gutter labels are pushed apart so none overlaps (sorted by y, 14 px
 *  apart, kept inside the pane). Pure; exported for the test. */
export function deoverlap(ys: number[], paneH: number, gap = LABEL_H): number[] {
  const order = ys.map((y, k) => ({ y, k })).sort((a, b) => a.y - b.y);
  const out = new Array<number>(ys.length);
  let prev = -Infinity;
  for (const o of order) { const y = Math.max(o.y, prev + gap); out[o.k] = y; prev = y; }
  // back-propagate from the bottom edge
  const maxY = paneH - gap / 2;
  let next = Infinity;
  for (let j = order.length - 1; j >= 0; j--) {
    const k = order[j].k;
    const y = Math.min(out[k], maxY, next - gap);
    out[k] = y; next = y;
  }
  return out.map((y) => Math.max(gap / 2, y));
}

export function draw(ctx: CanvasRenderingContext2D, f: Frame): DrawResult {
  const { w, h, bars, vp, theme } = f;
  const n = bars.length;
  const g = geometry(w, h, f.showVolume);
  ctx.clearRect(0, 0, w, h);
  ctx.font = `10.5px ${theme.font}`;
  ctx.textBaseline = "middle";
  ctx.lineWidth = 1;

  const fitPrices = f.levels.filter((l) => l.fit !== false && Number.isFinite(l.price)).map((l) => l.price);
  const range = priceRange(bars, vp, fitPrices);
  const decimals = priceDecimals(range, g.priceH);
  const y = (p: number) => yOf(p, range, g.priceH);

  // ── grid + price axis ──
  ctx.textAlign = "right";
  for (const p of priceTicks(range, g.priceH)) {
    const yy = Math.round(y(p)) + 0.5;
    if (yy < 0 || yy > g.priceH) continue;
    ctx.strokeStyle = theme.grid; ctx.beginPath(); ctx.moveTo(0, yy); ctx.lineTo(g.plotW, yy); ctx.stroke();
    ctx.fillStyle = theme.textFaint; ctx.fillText(fmtPrice(p, decimals), w - 4, yy);
  }
  // ── time axis ──
  ctx.textAlign = "center";
  for (const t of timeTicks(bars, vp, g.plotW, f.tf)) {
    const xx = Math.round(xOf(t.i + 0.5, vp, g.plotW)) + 0.5;
    if (xx < 0 || xx > g.plotW) continue;
    ctx.strokeStyle = t.major ? theme.gridStrong : theme.grid; ctx.beginPath(); ctx.moveTo(xx, 0); ctx.lineTo(xx, g.paneH); ctx.stroke();
    ctx.fillStyle = t.major ? theme.textDim : theme.textFaint;
    const tw = ctx.measureText(t.label).width;
    const lx = Math.min(Math.max(xx, tw / 2 + 2), g.plotW - tw / 2 - 2);
    ctx.fillText(t.label, lx, g.paneH + AXIS_B / 2 + 0.5);
  }

  // ── price pane (clipped) ──
  ctx.save();
  ctx.beginPath(); ctx.rect(0, 0, g.plotW, g.priceH); ctx.clip();
  // zones
  for (const z of f.zones) {
    const top = Number.isFinite(z.hi) ? y(z.hi) : 0, bot = Number.isFinite(z.lo) ? y(z.lo) : g.priceH;
    const y1 = Math.max(0, Math.min(top, bot)), y2 = Math.min(g.priceH, Math.max(top, bot));
    if (y2 - y1 <= 0) continue;
    ctx.fillStyle = withAlpha(z.color, z.alpha);
    ctx.fillRect(0, y1, g.plotW, y2 - y1);
    if (z.label && y2 - y1 > 16) { ctx.fillStyle = withAlpha(z.color, 0.75); ctx.textAlign = "left"; ctx.fillText(z.label, 6, y1 + 9); }
  }
  // candles
  const { first, last } = visibleIndexRange(vp, n);
  const perBar = pxPerBar(vp, g.plotW);
  const cw = Math.max(1, Math.min(perBar * 0.7, 40));
  for (let i = first; i <= last; i++) {
    const b = bars[i];
    const xc = Math.round(xOf(i + 0.5, vp, g.plotW)) + 0.5;
    const yo = y(b.open), yc = y(b.close), yh = y(b.high), yl = y(b.low);
    const col = b.close >= b.open ? theme.up : theme.down;
    ctx.strokeStyle = col; ctx.beginPath(); ctx.moveTo(xc, yh); ctx.lineTo(xc, yl); ctx.stroke();
    if (cw >= 2) { ctx.fillStyle = col; ctx.fillRect(xc - cw / 2, Math.min(yo, yc), cw, Math.max(1, Math.abs(yc - yo))); }
  }
  // markers
  for (const m of f.markers) {
    if (m.index < first - 1 || m.index > last + 1) continue;
    const xc = xOf(m.index + 0.5, vp, g.plotW), yy = y(m.price);
    ctx.fillStyle = m.color; ctx.beginPath();
    if (m.shape === "up") { ctx.moveTo(xc, yy + 5); ctx.lineTo(xc - 5, yy + 13); ctx.lineTo(xc + 5, yy + 13); }
    else { ctx.moveTo(xc, yy - 5); ctx.lineTo(xc - 5, yy - 13); ctx.lineTo(xc + 5, yy - 13); }
    ctx.closePath(); ctx.fill();
    if (m.label) { ctx.textAlign = "left"; ctx.fillStyle = m.color; ctx.fillText(m.label, xc + 8, m.shape === "up" ? yy + 9 : yy - 9); }
  }
  // level lines
  for (const l of f.levels) {
    if (!Number.isFinite(l.price)) continue;
    const yy = Math.round(y(l.price)) + 0.5;
    if (yy < -1 || yy > g.priceH + 1) continue;
    ctx.strokeStyle = l.muted ? withAlpha(l.color, 0.45) : l.color;
    ctx.setLineDash(DASHES[l.dash]); ctx.beginPath(); ctx.moveTo(0, yy); ctx.lineTo(g.plotW, yy); ctx.stroke(); ctx.setLineDash([]);
  }
  // last price
  const lastBar = n ? bars[n - 1] : null;
  if (lastBar) {
    const yy = Math.round(y(lastBar.close)) + 0.5;
    const col = n > 1 ? (lastBar.close >= bars[n - 2].close ? theme.up : theme.down) : theme.textDim;
    ctx.strokeStyle = withAlpha(col, 0.7); ctx.setLineDash([1, 3]); ctx.beginPath(); ctx.moveTo(0, yy); ctx.lineTo(g.plotW, yy); ctx.stroke(); ctx.setLineDash([]);
  }
  ctx.restore();

  // volume pane
  if (f.showVolume && g.volH > 4 && last >= first) {
    let maxVol = 0;
    for (let i = first; i <= last; i++) if (bars[i].volume > maxVol) maxVol = bars[i].volume;
    ctx.strokeStyle = theme.grid; ctx.beginPath(); ctx.moveTo(0, g.volTop + 0.5); ctx.lineTo(g.plotW, g.volTop + 0.5); ctx.stroke();
    if (maxVol > 0) for (let i = first; i <= last; i++) {
      const b = bars[i];
      const xc = xOf(i + 0.5, vp, g.plotW);
      const hh = (Math.max(0, b.volume) / maxVol) * (g.volH - 3);
      ctx.fillStyle = withAlpha(b.close >= b.open ? theme.up : theme.down, 0.32);
      ctx.fillRect(xc - cw / 2, g.volTop + g.volH - hh, Math.max(1, cw), hh);
    }
  }

  // ── gutter labels: levels (de-overlapped), then the last price ──
  const pill = (yy: number, text: string, color: string, alpha: number, textColor: string) => {
    ctx.fillStyle = withAlpha(color, alpha);
    ctx.fillRect(g.plotW + 2, yy - LABEL_H / 2, GUTTER_R - 3, LABEL_H);
    ctx.fillStyle = textColor; ctx.textAlign = "right"; ctx.fillText(text, w - 4, yy + 0.5);
  };
  const shown = f.levels.filter((l) => Number.isFinite(l.price));
  // the last price is pinned in the de-overlap set so no level pill ever
  // covers it (it is drawn last, on top)
  const lastY = lastBar ? y(lastBar.close) : null;
  const ys = deoverlap([...shown.map((l) => y(l.price)), ...(lastY !== null ? [lastY] : [])], g.priceH);
  shown.forEach((l, k) => {
    const yy = ys[k];
    if (yy < -LABEL_H || yy > g.priceH + LABEL_H) return;
    pill(yy, fmtPrice(l.price, decimals), l.color, l.muted ? 0.12 : 0.22, l.muted ? withAlpha(l.color, 0.75) : l.color);
  });
  if (lastBar && lastY !== null) {
    const col = n > 1 ? (lastBar.close >= bars[n - 2].close ? theme.up : theme.down) : theme.textDim;
    pill(ys[shown.length], fmtPrice(lastBar.close, decimals), col, 0.9, theme.surface);
  }
  // the levels' words sit inside the pane at the right edge, pushed apart
  // the same way and backed so they read over candles
  const inPane = shown.map((l, k) => ({ l, y: Math.round(y(l.price)) + 0.5, k })).filter((o) => o.y >= 0 && o.y <= g.priceH);
  const wordYs = deoverlap(inPane.map((o) => o.y - 8), g.priceH, 12);
  ctx.save(); ctx.beginPath(); ctx.rect(0, 0, g.plotW, g.priceH); ctx.clip();
  inPane.forEach((o, k) => {
    const wy = wordYs[k];
    const tw = ctx.measureText(o.l.label).width;
    ctx.fillStyle = withAlpha(theme.surface, 0.72);
    ctx.fillRect(g.plotW - tw - 8, wy - 6, tw + 6, 12);
    ctx.textAlign = "right"; ctx.fillStyle = o.l.muted ? withAlpha(o.l.color, 0.7) : o.l.color;
    ctx.fillText(o.l.label, g.plotW - 5, wy + 0.5);
  });
  ctx.restore();

  // ── crosshair ──
  let crossIndex: number | null = null, crossPrice: number | null = null;
  if (f.cross && n) {
    const i = indexAtX(f.cross.x, vp, g.plotW, n);
    if (i !== null && f.cross.x >= 0 && f.cross.x <= g.plotW) {
      crossIndex = i;
      const xc = Math.round(xOf(i + 0.5, vp, g.plotW)) + 0.5;
      ctx.strokeStyle = withAlpha(theme.textDim, 0.55); ctx.setLineDash([3, 3]);
      ctx.beginPath(); ctx.moveTo(xc, 0); ctx.lineTo(xc, g.paneH); ctx.stroke();
      if (f.cross.y >= 0 && f.cross.y <= g.priceH) {
        const yy = Math.round(f.cross.y) + 0.5;
        ctx.beginPath(); ctx.moveTo(0, yy); ctx.lineTo(g.plotW, yy); ctx.stroke();
        crossPrice = priceOfY(f.cross.y, range, g.priceH);
        ctx.setLineDash([]);
        pill(yy, fmtPrice(crossPrice, decimals), theme.text, 0.92, theme.surface);
      }
      ctx.setLineDash([]);
      const label = fmtBarTime(bars[i].time, f.tf);
      const tw = ctx.measureText(label).width + 10;
      const lx = Math.min(Math.max(xc - tw / 2, 0), g.plotW - tw);
      ctx.fillStyle = withAlpha(theme.text, 0.92); ctx.fillRect(lx, g.paneH + 1, tw, AXIS_B - 2);
      ctx.fillStyle = theme.surface; ctx.textAlign = "center"; ctx.fillText(label, lx + tw / 2, g.paneH + AXIS_B / 2 + 0.5);
    }
  }
  return { range, decimals, crossIndex, crossPrice };
}
