// Bot page chart — scales (pure). Bar index ↔ x, price ↔ y, nice ticks,
// the viewport (a fractional bar-index window) and its zoom / pan rules.
// No DOM; pinned by scales.test.ts. The renderer (draw.ts) and the
// controller (BotChart.tsx) only call these.
//
// Index space: bar i occupies [i, i + 1); its centre is i + 0.5. The
// viewport is [from, to) in that space and may run past either end (that
// is the right-hand padding after the last bar).

export interface Bar { time: number; open: number; high: number; low: number; close: number; volume: number }
export interface Viewport { from: number; to: number }
export interface PriceRange { lo: number; hi: number }

export type Timeframe = "1Min" | "5Min" | "15Min" | "1Hour" | "1Day";
export interface TimeframeSpec { tf: Timeframe; label: string; minutes: number; limit: number; show: number }
/** `limit` = bars requested from /api/bot/bars; `show` = bars in the fitted
 *  view (the rest is history the user can pan into). */
export const TIMEFRAMES: readonly TimeframeSpec[] = [
  // 1m (2026-10-22, human-asked): a full session is 390 bars; the IEX feed
  // can leave gaps in thin names — the empty state says so, nothing is filled
  { tf: "1Min", label: "1m", minutes: 1, limit: 390, show: 120 },
  { tf: "5Min", label: "5m", minutes: 5, limit: 240, show: 110 },
  { tf: "15Min", label: "15m", minutes: 15, limit: 240, show: 100 },
  { tf: "1Hour", label: "1h", minutes: 60, limit: 300, show: 120 },
  { tf: "1Day", label: "1D", minutes: 1440, limit: 260, show: 120 },
];
export const DEFAULT_TIMEFRAME: Timeframe = "1Hour";
export function timeframeSpec(tf: string | null | undefined): TimeframeSpec {
  return TIMEFRAMES.find((t) => t.tf === tf) ?? TIMEFRAMES.find((t) => t.tf === DEFAULT_TIMEFRAME)!;
}

/** Hard caps (Law IV): the chart never holds more than MAX_BARS; a longer
 *  series keeps its newest bars. */
export const MAX_BARS = 600;
export const MIN_VISIBLE_BARS = 8;
export const RIGHT_PAD_FRAC = 0.08;

export function capBars(bars: readonly Bar[]): Bar[] {
  const clean = bars.filter((b) => b && [b.time, b.open, b.high, b.low, b.close].every((v) => Number.isFinite(v)));
  clean.sort((a, b) => a.time - b.time);
  return clean.length > MAX_BARS ? clean.slice(clean.length - MAX_BARS) : clean;
}

export function spanOf(vp: Viewport): number { return vp.to - vp.from; }

/** The fitted view: the last `show` bars with a little room on the right. */
export function fitViewport(n: number, show: number): Viewport {
  const span = Math.max(MIN_VISIBLE_BARS, Math.min(Math.max(show, MIN_VISIBLE_BARS), Math.max(n, MIN_VISIBLE_BARS)));
  const pad = span * RIGHT_PAD_FRAC;
  return { from: n - span, to: n + pad };
}

/** Keep at least MIN_VISIBLE_BARS bars inside the window at either end. */
export function clampViewport(vp: Viewport, n: number): Viewport {
  const span = Math.max(MIN_VISIBLE_BARS, Math.min(spanOf(vp), Math.max(n, MIN_VISIBLE_BARS) * 1.5));
  const minFrom = -(span - MIN_VISIBLE_BARS);
  const maxFrom = Math.max(minFrom, n - MIN_VISIBLE_BARS);
  const from = Math.min(Math.max(vp.from, minFrom), maxFrom);
  return { from, to: from + span };
}

export function xOf(i: number, vp: Viewport, w: number): number { return ((i - vp.from) / spanOf(vp)) * w; }
export function iOf(x: number, vp: Viewport, w: number): number { return vp.from + (x / w) * spanOf(vp); }
export function pxPerBar(vp: Viewport, w: number): number { return w / spanOf(vp); }

/** Zoom by `factor` (< 1 = in) keeping the bar under `anchorI` fixed. */
export function zoomViewport(vp: Viewport, factor: number, anchorI: number, n: number): Viewport {
  if (!Number.isFinite(factor) || factor <= 0) return vp;
  const span = spanOf(vp);
  const next = Math.max(MIN_VISIBLE_BARS, Math.min(span * factor, Math.max(n, MIN_VISIBLE_BARS) * 1.5));
  const f = next / span;
  const from = anchorI - (anchorI - vp.from) * f;
  return clampViewport({ from, to: from + next }, n);
}

export function panViewport(vp: Viewport, dBars: number, n: number): Viewport {
  return clampViewport({ from: vp.from + dBars, to: vp.to + dBars }, n);
}

/** When `added` new bars arrive at the right while the view is on the live
 *  edge, follow them; a view panned back into history stays put. */
export function followViewport(vp: Viewport, nBefore: number, nAfter: number): Viewport {
  const added = nAfter - nBefore;
  if (added <= 0) return vp;
  const onLiveEdge = vp.to >= nBefore - 0.5;
  return onLiveEdge ? { from: vp.from + added, to: vp.to + added } : vp;
}

export function visibleIndexRange(vp: Viewport, n: number): { first: number; last: number } {
  const first = Math.max(0, Math.floor(vp.from));
  const last = Math.min(n - 1, Math.ceil(vp.to) - 1);
  return { first, last };
}

/** Price range of the visible bars plus any level that should stay in view,
 *  padded. With no bars the levels alone set it; with nothing at all, 0–1. */
export function priceRange(bars: readonly Bar[], vp: Viewport, levels: readonly number[] = [], padFrac = 0.07): PriceRange {
  let lo = Infinity, hi = -Infinity;
  const { first, last } = visibleIndexRange(vp, bars.length);
  for (let i = first; i <= last; i++) { const b = bars[i]; if (b.low < lo) lo = b.low; if (b.high > hi) hi = b.high; }
  for (const p of levels) if (Number.isFinite(p)) { if (p < lo) lo = p; if (p > hi) hi = p; }
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return { lo: 0, hi: 1 };
  if (hi === lo) { const d = Math.max(Math.abs(hi) * 0.01, 0.01); return { lo: lo - d, hi: hi + d }; }
  const pad = (hi - lo) * padFrac;
  return { lo: lo - pad, hi: hi + pad };
}

export function yOf(price: number, r: PriceRange, h: number): number { return h - ((price - r.lo) / (r.hi - r.lo)) * h; }
export function priceOfY(y: number, r: PriceRange, h: number): number { return r.lo + ((h - y) / h) * (r.hi - r.lo); }

/** The 1 / 2 / 2.5 / 5 × 10^k step that yields about `target` ticks. */
export function niceStep(span: number, target: number): number {
  if (!(span > 0) || !(target > 0)) return 1;
  const raw = span / target;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  for (const m of [1, 2, 2.5, 5, 10]) if (m * mag >= raw) return m * mag;
  return 10 * mag;
}

export function priceTicks(r: PriceRange, h: number, minPx = 30): number[] {
  const target = Math.max(2, Math.floor(h / minPx));
  const step = niceStep(r.hi - r.lo, target);
  const out: number[] = [];
  const start = Math.ceil(r.lo / step) * step;
  for (let p = start; p <= r.hi + step * 1e-9 && out.length < 200; p += step) out.push(Math.round(p / step) * step);
  return out;
}

export function decimalsFor(step: number): number {
  if (!(step > 0)) return 2;
  const s = step.toFixed(10).replace(/0+$/, "").replace(/\.$/, "");
  const dot = s.indexOf(".");
  return dot < 0 ? 0 : s.length - dot - 1;
}

/** Price label decimals: at least what the tick step needs, 2 for sub-$1000
 *  prices, 0 above $10 000 (the pill stays narrow). */
export function priceDecimals(r: PriceRange, h: number): number {
  const step = niceStep(r.hi - r.lo, Math.max(2, Math.floor(h / 30)));
  const base = r.hi >= 10_000 ? 0 : r.hi >= 1000 ? 1 : 2;
  return Math.max(base, decimalsFor(step));
}

export function fmtPrice(p: number, decimals: number): string {
  return Number.isFinite(p) ? p.toLocaleString("en-US", { minimumFractionDigits: decimals, maximumFractionDigits: decimals }) : "—";
}

// ── time ────────────────────────────────────────────────────────────────────

export interface EtParts { date: string; hm: string; minuteOfDay: number; dayNum: number; monthShort: string; year: number; weekdayShort: string }
const ET_FMT = new Intl.DateTimeFormat("en-US", { timeZone: "America/New_York", year: "numeric", month: "short", day: "numeric", weekday: "short", hour: "2-digit", minute: "2-digit", hour12: false });
export function etParts(timeS: number): EtParts {
  const parts = ET_FMT.formatToParts(new Date(timeS * 1000));
  const g = (t: string) => parts.find((p) => p.type === t)?.value ?? "";
  const hour = Number(g("hour")) % 24, minute = Number(g("minute"));
  const month = g("month"), day = Number(g("day")), year = Number(g("year"));
  const mi = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"].indexOf(month);
  const date = `${year}-${String(mi + 1).padStart(2, "0")}-${String(day).padStart(2, "0")}`;
  return { date, hm: `${String(hour).padStart(2, "0")}:${String(minute).padStart(2, "0")}`, minuteOfDay: hour * 60 + minute, dayNum: day, monthShort: month, year, weekdayShort: g("weekday") };
}

export interface TimeTick { i: number; label: string; major: boolean }

/** Ticks along the visible bars: for intraday frames a minute step chosen so
 *  ticks sit at least `minPx` apart, with a day boundary as a major tick;
 *  for the daily frame a month boundary is major and day numbers fill in. */
export function timeTicks(bars: readonly Bar[], vp: Viewport, w: number, tf: Timeframe, minPx = 72): TimeTick[] {
  const n = bars.length;
  if (!n || w <= 0) return [];
  const { first, last } = visibleIndexRange(vp, n);
  if (last < first) return [];
  const perBar = pxPerBar(vp, w);
  const barsPerTick = Math.max(1, Math.ceil(minPx / Math.max(perBar, 1e-6)));
  const spec = timeframeSpec(tf);
  const out: TimeTick[] = [];
  let prev: EtParts | null = first > 0 ? etParts(bars[first - 1].time) : null;
  let lastTickI = -Infinity;
  if (tf === "1Day") {
    for (let i = first; i <= last; i++) {
      const p = etParts(bars[i].time);
      const newMonth = !prev || p.monthShort !== prev.monthShort || p.year !== prev.year;
      if (newMonth) {
        while (out.length && !out[out.length - 1].major && i - out[out.length - 1].i < barsPerTick) out.pop();
        out.push({ i, label: !prev || p.year !== prev.year ? `${p.monthShort} ${p.year}` : p.monthShort, major: true });
        lastTickI = i;
      } else if (i - lastTickI >= barsPerTick && i - first >= Math.min(barsPerTick, 2)) {
        out.push({ i, label: String(p.dayNum), major: false });
        lastTickI = i;
      }
      prev = p;
    }
    return out;
  }
  // the smallest round minute step that keeps ticks `minPx` apart; a tick
  // lands on the first bar past each step boundary, so bars aligned to :30
  // still get their hourly marks
  const steps = [5, 10, 15, 30, 60, 120, 240, 390];
  const minMinutes = barsPerTick * spec.minutes;
  const step = steps.find((s) => s >= minMinutes && s % spec.minutes === 0) ?? 390;
  for (let i = first; i <= last; i++) {
    const p = etParts(bars[i].time);
    const newDay = !prev || p.date !== prev.date;
    if (newDay) {
      // a day label wins over a minute tick it would sit on top of
      while (out.length && !out[out.length - 1].major && i - out[out.length - 1].i < barsPerTick) out.pop();
      out.push({ i, label: `${p.weekdayShort} ${p.dayNum}`, major: true });
      lastTickI = i;
    } else if (prev && Math.floor(p.minuteOfDay / step) > Math.floor(prev.minuteOfDay / step) && i - lastTickI >= barsPerTick) {
      out.push({ i, label: p.hm, major: false });
      lastTickI = i;
    }
    prev = p;
  }
  return out;
}

/** The bar under pixel x, or null outside the series. */
export function indexAtX(x: number, vp: Viewport, w: number, n: number): number | null {
  const i = Math.floor(iOf(x, vp, w));
  return i >= 0 && i < n ? i : null;
}

export function fmtBarTime(timeS: number, tf: Timeframe): string {
  const p = etParts(timeS);
  return tf === "1Day" ? `${p.weekdayShort} ${p.monthShort} ${p.dayNum}, ${p.year}` : `${p.weekdayShort} ${p.monthShort} ${p.dayNum} · ${p.hm} ET`;
}
/** The same instant, short enough for the readout row ("through 15:55 ET" / "through Oct 21"). */
export function fmtBarTimeShort(timeS: number, tf: Timeframe): string {
  const p = etParts(timeS);
  return tf === "1Day" ? `${p.monthShort} ${p.dayNum}` : `${p.hm} ET`;
}
