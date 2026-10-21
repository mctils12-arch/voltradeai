// Bot page chart — the interactive canvas component. Owns ONE canvas and a
// controller registered on the page's frame loop (frameCore: one rAF loop
// per view, Rendering & Motion Law I). Pointer and wheel input set TARGETS
// on two critically damped springs (the viewport's edges); the loop
// interpolates and the renderer draws only on dirty frames. Pointer Events
// only; wheel and pinch share zoomInput's normalisation. Teardown detaches
// the springs, unregisters the frame callback and drops the observers.
import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { useQuery } from "@tanstack/react-query";
import { apiRequest } from "@/lib/queryClient";
import { frameCore, PRIORITY, SPRING, type Spring } from "@/render/frameCore";
import { wheelToLogDelta, spread, centroid } from "@/render/zoomInput";
import {
  type Bar, type Viewport, type Timeframe, TIMEFRAMES, timeframeSpec, capBars, fitViewport, clampViewport, zoomViewport, panViewport, followViewport,
  iOf, pxPerBar, fmtBarTime, fmtPrice,
} from "./scales";
import { draw, readTheme, geometry, FALLBACK_THEME, type Theme, type DrawLevel, type DrawZone, type DrawMarker } from "./draw";
import { overlayFor, type OverlaySubject, type Overlay } from "./levels";
import { BOT_READ_TIMEOUT_MS } from "../useBotData";
import { fmtAge, MINUS } from "../format";

// ── bars ────────────────────────────────────────────────────────────────────
interface BarsBody { bars?: Array<Partial<Bar>>; ticker?: string; timeframe?: string; error?: string }
export interface BarsResult { bars: Bar[]; fetchedAt: number }

/** `apiRequest` throws "502: {"error":"…"}" for a non-OK route; show the
 *  message, not the JSON. */
export function humanError(e: unknown): string {
  const m = e instanceof Error ? e.message : String(e);
  const json = /^\d{3}:\s*(\{.*\})\s*$/s.exec(m);
  const short = m.length > 160 ? m.slice(0, 157) + "…" : m;
  if (!json) return short;
  try {
    const b = JSON.parse(json[1]) as { error?: unknown };
    return typeof b.error === "string" && b.error ? b.error : short;
  } catch {
    return short; // not JSON after all: the raw text is the message
  }
}

export async function fetchBars(symbol: string, tf: Timeframe, limit: number): Promise<BarsResult> {
  const r = await apiRequest("GET", `/api/bot/bars/${encodeURIComponent(symbol)}?timeframe=${tf}&limit=${limit}`, undefined, { timeoutMs: BOT_READ_TIMEOUT_MS });
  const body = (await r.json()) as BarsBody;
  if (typeof body.error === "string" && body.error) throw new Error(body.error);
  const bars = capBars((body.bars ?? []).map((b) => ({ time: Number(b.time), open: Number(b.open), high: Number(b.high), low: Number(b.low), close: Number(b.close), volume: Number(b.volume ?? 0) })));
  return { bars, fetchedAt: Date.now() };
}

export const BARS_REFRESH_MS = 30_000;
/** `paused` = the panel is held: the display freezes while the bot and
 *  every other subscription continue (no refetch, cached bars shown). */
export function useBars(symbol: string | null, tf: Timeframe, paused = false) {
  const spec = timeframeSpec(tf);
  return useQuery<BarsResult>({
    queryKey: ["/api/bot/bars", symbol, tf],
    queryFn: () => fetchBars(symbol as string, tf, spec.limit),
    enabled: !!symbol,
    refetchInterval: paused ? false : BARS_REFRESH_MS,
    staleTime: 20_000,
    retry: 1,
  });
}

// ── controller (no React inside) ────────────────────────────────────────────
export interface ChartOverlay { levels: DrawLevel[]; zones: DrawZone[]; markers: DrawMarker[] }
const EMPTY_OVERLAY: ChartOverlay = { levels: [], zones: [], markers: [] };
/** Device pixel ratio cap: the S24 reports 3; textures and fill cost scale
 *  with its square (Rendering & Motion Law harness note: "DPR 3 capped to 2"). */
export const DPR_CAP = 2;

export class ChartController {
  bars: Bar[] = [];
  n = 0;
  size = { w: 0, h: 0, dpr: 1 };
  cross: { x: number; y: number } | null = null;
  overlay: ChartOverlay = EMPTY_OVERLAY;
  tf: Timeframe = "1Hour";
  theme: Theme = FALLBACK_THEME;
  showVolume = true;
  onCross: ((i: number | null, price: number | null) => void) | null = null;
  private from: Spring;
  private to: Spring;
  private detachFrom: () => void;
  private detachTo: () => void;
  private unregister: () => void;
  private target: Viewport = { from: 0, to: 8 };
  private fitted = false;
  private dirty = true;
  private lastCrossIndex: number | null = null;
  private disposed = false;

  constructor(private canvas: HTMLCanvasElement, theme: Theme) {
    this.theme = theme;
    const loop = frameCore();
    const a = loop.createSpring(0, { stiffness: SPRING.PAN_STIFFNESS, damping: SPRING.PAN_DAMPING });
    const b = loop.createSpring(8, { stiffness: SPRING.PAN_STIFFNESS, damping: SPRING.PAN_DAMPING });
    this.from = a.spring; this.to = b.spring; this.detachFrom = a.detach; this.detachTo = b.detach;
    this.unregister = loop.register(() => this.frame(), PRIORITY.RENDER, { label: "bot-chart" });
  }

  get viewport(): Viewport { return { from: this.from.value, to: this.to.value }; }
  get targetViewport(): Viewport { return this.target; }
  plotW(): number { return geometry(this.size.w, this.size.h, this.showVolume).plotW; }

  setSize(w: number, h: number, dpr: number): void {
    const d = Math.min(DPR_CAP, Math.max(1, dpr || 1));
    if (w === this.size.w && h === this.size.h && d === this.size.dpr) return;
    this.size = { w, h, dpr: d };
    this.canvas.width = Math.max(1, Math.round(w * d));
    this.canvas.height = Math.max(1, Math.round(h * d));
    if (!this.fitted && this.n) this.fit(true);
    this.dirty = true;
  }
  setTf(tf: Timeframe): void { if (tf !== this.tf) { this.tf = tf; this.fitted = false; } }
  setOverlay(o: ChartOverlay): void { this.overlay = o; this.dirty = true; }
  setTheme(t: Theme): void { this.theme = t; this.dirty = true; }

  /** New bars: refit on a new series; otherwise follow the live edge. */
  setBars(bars: Bar[], refit: boolean): void {
    const before = this.n;
    this.bars = bars; this.n = bars.length;
    if (refit || !this.fitted) this.fit(true);
    else this.setTarget(clampViewport(followViewport(this.target, before, this.n), this.n), true);
    this.dirty = true;
  }
  fit(snap: boolean): void {
    this.setTarget(fitViewport(this.n, timeframeSpec(this.tf).show), snap);
    this.fitted = this.n > 0;
  }
  private setTarget(vp: Viewport, snap = false): void {
    this.target = vp;
    if (snap) { this.from.snap(vp.from); this.to.snap(vp.to); } else { this.from.setTarget(vp.from); this.to.setTarget(vp.to); }
    this.dirty = true;
  }
  /** Pan from a gesture's starting viewport by a pixel delta. */
  panByPx(dx: number, base: Viewport): void {
    if (!this.n) return;
    this.setTarget(panViewport(base, -dx / pxPerBar(base, this.plotW()), this.n));
  }
  panByBars(dBars: number): void { if (this.n) this.setTarget(panViewport(this.target, dBars, this.n)); }
  /** Zoom around pixel x by a log delta (positive = out), anchored on the
   *  TARGET viewport so successive wheel events compose. */
  zoomAtPx(x: number, logDelta: number): void {
    if (!this.n || !logDelta) return;
    const anchor = iOf(x, this.target, this.plotW());
    this.setTarget(zoomViewport(this.target, Math.exp(logDelta), anchor, this.n));
  }
  /** Pinch: scale the gesture's starting viewport around its starting
   *  centroid, then pan by the centroid's travel. */
  pinch(base: Viewport, cx0: number, cx: number, scale: number): void {
    if (!this.n || !(scale > 0)) return;
    const w = this.plotW();
    const vp = zoomViewport(base, 1 / scale, iOf(cx0, base, w), this.n);
    this.setTarget(panViewport(vp, -(cx - cx0) / pxPerBar(vp, w), this.n));
  }
  setCross(c: { x: number; y: number } | null): void {
    if ((c === null) !== (this.cross === null) || (c && this.cross && (c.x !== this.cross.x || c.y !== this.cross.y))) { this.cross = c; this.dirty = true; }
  }

  private frame(): void {
    if (this.disposed) return;
    const awake = !this.from.asleep || !this.to.asleep;
    if (!this.dirty && !awake) return;
    const { w, h, dpr } = this.size;
    if (!w || !h) return;
    const ctx = this.canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const res = draw(ctx, { bars: this.bars, vp: this.viewport, w, h, tf: this.tf, theme: this.theme, levels: this.overlay.levels, zones: this.overlay.zones, markers: this.overlay.markers, cross: this.cross, showVolume: this.showVolume });
    this.dirty = awake;
    if (res.crossIndex !== this.lastCrossIndex) { this.lastCrossIndex = res.crossIndex; this.onCross?.(res.crossIndex, res.crossPrice); }
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.unregister(); this.detachFrom(); this.detachTo();
    this.bars = []; this.n = 0; this.overlay = EMPTY_OVERLAY; this.onCross = null;
  }
}

/** Index of the bar that contains `timeS` (bars are `minutes` wide). */
export function barIndexAt(bars: readonly Bar[], timeS: number, minutes: number): number | null {
  if (!bars.length || !Number.isFinite(timeS)) return null;
  const width = minutes * 60;
  for (let i = bars.length - 1; i >= 0; i--) if (bars[i].time <= timeS) return timeS < bars[i].time + width * 1.5 ? i : null;
  return null;
}

// ── the component ───────────────────────────────────────────────────────────
export const LOADING_WATCHDOG_MS = 30_000;

export interface BotChartProps {
  /** what to fetch bars for (an option leg charts its underlying) */
  symbol: string | null;
  subject: OverlaySubject | null;
  tf: Timeframe;
  onTf: (tf: Timeframe) => void;
  /** the symbol strip (its own wrapping row above the chart) */
  top?: ReactNode;
  /** held panel: no refetch */
  paused?: boolean;
}

export function BotChart({ symbol, subject, tf, onTf, top, paused = false }: BotChartProps) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const ctrlRef = useRef<ChartController | null>(null);
  const [theme, setTheme] = useState<Theme>(FALLBACK_THEME);
  const [cross, setCross] = useState<{ i: number | null; price: number | null }>({ i: null, price: null });
  const [now, setNow] = useState(() => Date.now());
  const q = useBars(symbol, tf, paused);
  const bars = q.data?.bars ?? null;

  // the controller lives for the component's life; the canvas is stable
  useEffect(() => {
    const canvas = canvasRef.current, wrap = wrapRef.current;
    if (!canvas || !wrap) return;
    const th = readTheme(wrap);
    setTheme(th);
    const c = new ChartController(canvas, th);
    ctrlRef.current = c;
    c.onCross = (i, price) => setCross({ i, price });
    const ro = new ResizeObserver(() => { const r = wrap.getBoundingClientRect(); c.setSize(Math.round(r.width), Math.round(r.height), window.devicePixelRatio || 1); });
    ro.observe(wrap);
    const r0 = wrap.getBoundingClientRect();
    c.setSize(Math.round(r0.width), Math.round(r0.height), window.devicePixelRatio || 1);

    // ── input (Pointer Events only) ──
    const pointers = new Map<number, { x: number; y: number }>();
    let drag: { x0: number; base: Viewport; moved: boolean } | null = null;
    let pinch: { spread0: number; cx0: number; base: Viewport } | null = null;
    const local = (e: PointerEvent) => { const r = canvas.getBoundingClientRect(); return { x: e.clientX - r.left, y: e.clientY - r.top }; };
    const onDown = (e: PointerEvent) => {
      if (e.button !== 0 && e.pointerType === "mouse") return;
      canvas.setPointerCapture(e.pointerId);
      const p = local(e);
      pointers.set(e.pointerId, p);
      if (pointers.size === 1) { drag = { x0: p.x, base: c.targetViewport, moved: false }; pinch = null; }
      else if (pointers.size === 2) {
        const [a, b] = [...pointers.values()];
        pinch = { spread0: Math.max(1, spread(a, b)), cx0: centroid(a, b).x, base: c.targetViewport }; drag = null;
      }
      if (e.pointerType === "touch") c.setCross(p);
    };
    const onMove = (e: PointerEvent) => {
      const p = local(e);
      if (pointers.has(e.pointerId)) pointers.set(e.pointerId, p);
      if (pinch && pointers.size === 2) {
        const [a, b] = [...pointers.values()];
        c.pinch(pinch.base, pinch.cx0, centroid(a, b).x, spread(a, b) / pinch.spread0);
        c.setCross(null);
        return;
      }
      if (drag) {
        const dx = p.x - drag.x0;
        if (Math.abs(dx) > 2) drag.moved = true;
        c.panByPx(dx, drag.base);
        if (e.pointerType !== "mouse") { c.setCross(null); return; }
      }
      c.setCross(p);
    };
    const onUp = (e: PointerEvent) => {
      pointers.delete(e.pointerId);
      if (canvas.hasPointerCapture(e.pointerId)) canvas.releasePointerCapture(e.pointerId);
      if (pointers.size < 2) pinch = null;
      if (pointers.size === 0) drag = null;
    };
    const onLeave = () => { if (!drag && !pinch) c.setCross(null); };
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const r = canvas.getBoundingClientRect();
      if (Math.abs(e.deltaX) > Math.abs(e.deltaY)) { c.panByPx(-e.deltaX, c.targetViewport); return; }
      c.zoomAtPx(e.clientX - r.left, wheelToLogDelta(e.deltaY, e.deltaMode, r.height || 800));
    };
    const onDbl = () => c.fit(false);
    const onKey = (e: KeyboardEvent) => {
      const span = c.targetViewport.to - c.targetViewport.from;
      const cx = c.plotW() / 2;
      if (e.key === "ArrowLeft") { c.panByBars(-span * 0.1); e.preventDefault(); }
      else if (e.key === "ArrowRight") { c.panByBars(span * 0.1); e.preventDefault(); }
      else if (e.key === "+" || e.key === "=") { c.zoomAtPx(cx, -0.25); e.preventDefault(); }
      else if (e.key === "-" || e.key === "_") { c.zoomAtPx(cx, 0.25); e.preventDefault(); }
      else if (e.key === "Home" || e.key === "0") { c.fit(false); e.preventDefault(); }
    };
    canvas.addEventListener("pointerdown", onDown);
    canvas.addEventListener("pointermove", onMove);
    canvas.addEventListener("pointerup", onUp);
    canvas.addEventListener("pointercancel", onUp);
    canvas.addEventListener("pointerleave", onLeave);
    canvas.addEventListener("wheel", onWheel, { passive: false });
    canvas.addEventListener("dblclick", onDbl);
    canvas.addEventListener("keydown", onKey);
    return () => {
      ro.disconnect();
      canvas.removeEventListener("pointerdown", onDown);
      canvas.removeEventListener("pointermove", onMove);
      canvas.removeEventListener("pointerup", onUp);
      canvas.removeEventListener("pointercancel", onUp);
      canvas.removeEventListener("pointerleave", onLeave);
      canvas.removeEventListener("wheel", onWheel);
      canvas.removeEventListener("dblclick", onDbl);
      canvas.removeEventListener("keydown", onKey);
      c.dispose();
      ctrlRef.current = null;
    };
  }, []);

  // a new symbol or timeframe refits; a refresh of the same series follows the live edge
  const seriesKey = `${symbol ?? ""}|${tf}`;
  const lastSeries = useRef<string>("");
  useEffect(() => {
    const c = ctrlRef.current; if (!c) return;
    c.setTf(tf);
    const refit = lastSeries.current !== seriesKey;
    lastSeries.current = seriesKey;
    c.setBars(bars ?? [], refit);
  }, [bars, seriesKey, tf]);

  const overlay: Overlay = useMemo(() => (subject ? overlayFor(subject, theme) : { levels: [], zones: [], legend: [], notes: [] }), [subject, theme]);
  useEffect(() => {
    const c = ctrlRef.current; if (!c) return;
    const markers: DrawMarker[] = [];
    const lv = subject?.levels;
    if (bars && subject && !subject.isOption && lv?.heldMinutes != null && subject.entryPrice !== null && Number.isFinite(subject.entryPrice)) {
      const entryS = (Date.now() - lv.heldMinutes * 60_000) / 1000;
      const i = barIndexAt(bars, entryS, timeframeSpec(tf).minutes);
      if (i !== null) markers.push({ index: i, price: subject.entryPrice, color: theme.accent, shape: "up", label: "in" });
    }
    c.setOverlay({ levels: overlay.levels, zones: overlay.zones, markers });
  }, [overlay, bars, subject, theme, tf]);

  // clock for the age line and the loading watchdog
  useEffect(() => { const id = setInterval(() => setNow(Date.now()), 1_000); return () => clearInterval(id); }, []);
  const [loadingSince, setLoadingSince] = useState<number | null>(null);
  useEffect(() => { setLoadingSince(q.isLoading ? Date.now() : null); }, [q.isLoading, seriesKey]);
  const stuck = q.isLoading && loadingSince !== null && now - loadingSince > LOADING_WATCHDOG_MS;

  const last = bars && bars.length ? bars[bars.length - 1] : null;
  const prev = bars && bars.length > 1 ? bars[bars.length - 2] : null;
  const shown = cross.i !== null && bars && bars[cross.i] ? bars[cross.i] : last;
  const dec = shown ? (shown.close < 2 ? 4 : 2) : 2;
  const chg = shown && prev && cross.i === null ? ((shown.close - prev.close) / prev.close) * 100 : shown && cross.i !== null && cross.i > 0 && bars ? ((shown.close - bars[cross.i - 1].close) / bars[cross.i - 1].close) * 100 : null;

  const vol = shown ? (shown.volume >= 1e6 ? `${(shown.volume / 1e6).toFixed(1)}M` : shown.volume >= 1e3 ? `${Math.round(shown.volume / 1e3)}k` : String(Math.round(shown.volume))) : "";
  return (
    <div className="vt-chart" data-symbol={symbol ?? undefined}>
      {top}
      <div className="vt-chart-top">
        <div className="vt-chart-readout" aria-live="off">
          {shown ? (
            <>
              <span className="vt-bot-t3">{cross.i !== null ? fmtBarTime(shown.time, tf) : `${symbol} · last`}</span>
              <span>O {fmtPrice(shown.open, dec)}</span><span>H {fmtPrice(shown.high, dec)}</span><span>L {fmtPrice(shown.low, dec)}</span>
              <span className="vt-bot-t1">C {fmtPrice(shown.close, dec)}</span>
              {chg !== null ? <span className={chg < 0 ? "vt-bot-r" : "vt-bot-g"}>{chg < 0 ? MINUS : "+"}{Math.abs(chg).toFixed(2)}%</span> : null}
              <span className="vt-bot-t3">vol {vol}</span>
              {cross.i === null && q.data && last ? <span className="vt-bot-t3" title={`bars through ${fmtBarTime(last.time, tf)}`}>fetched {fmtAge(Math.max(0, now - q.data.fetchedAt))} ago</span> : null}
            </>
          ) : <span className="vt-bot-t3">{symbol ? `${symbol} · ${timeframeSpec(tf).label}` : "no symbol"}</span>}
        </div>
        <div className="vt-chart-tf" role="group" aria-label="Timeframe">
          {TIMEFRAMES.map((t) => <button key={t.tf} type="button" aria-pressed={t.tf === tf} onClick={() => onTf(t.tf)}>{t.label}</button>)}
        </div>
      </div>
      <div className="vt-chart-wrap" ref={wrapRef}>
        <canvas ref={canvasRef} tabIndex={0} role="img" aria-label={symbol ? `${symbol} ${timeframeSpec(tf).label} candles with the bot's levels; drag to pan, wheel or pinch to zoom, double-click to fit` : "chart"} data-vt-chart={symbol ?? ""} />
        {!symbol ? <div className="vt-chart-state"><span className="vt-bot-st vt-bot-t3"><span className="vt-bot-dot" data-out="true" />nothing to chart</span></div> : null}
        {symbol && q.isLoading ? <><div className="vt-chart-skel" aria-hidden="true" /><div className="vt-chart-state"><span className="vt-bot-st"><span className="vt-bot-dot" />loading {symbol} · {timeframeSpec(tf).label}</span>{stuck ? <span className="vt-chart-note">still fetching after 30 s · retrying · the bars route is slow or down</span> : null}</div></> : null}
        {symbol && q.isError ? <div className="vt-chart-state"><span className="vt-bot-st vt-bot-r"><span className="vt-bot-dot" />bars unavailable</span><span className="vt-chart-note">{humanError(q.error)}</span><button type="button" className="vt-bot-wbtn" onClick={() => { void q.refetch(); }}>retry</button></div> : null}
        {symbol && q.isSuccess && (!bars || !bars.length) ? <div className="vt-chart-state"><span className="vt-bot-st vt-bot-t3"><span className="vt-bot-dot" data-out="true" />no bars for {symbol} at {timeframeSpec(tf).label}</span><span className="vt-chart-note">the data feed returned nothing for this timeframe · try another</span></div> : null}
      </div>
      {overlay.legend.length || overlay.notes.length ? (
        <div className="vt-chart-legend">
          {overlay.legend.map(([k, v]) => <span key={k}><span className="k">{k}</span><span className="v">{v}</span></span>)}
          {overlay.notes.map((n, k) => <span key={`n${k}`} className="vt-chart-note">{n}</span>)}
        </div>
      ) : null}
    </div>
  );
}
