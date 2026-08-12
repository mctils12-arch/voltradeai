// perfHud — the `?perf=1` developer overlay.
//
// Built FIRST in the rendering overhaul, deliberately: none of the Laws are
// verifiable by looking at the map. "The moon looks fuzzy" is not a
// measurement; "p95 frame time 41ms, 312 tiles resident, 190MB VRAM, 14
// requests in flight" is. Every later PR in the overhaul is checked against
// this readout.
//
// It is a development instrument, not a product surface: it renders only
// when the URL carries `?perf=1` (or localStorage `vt:perf`), and it is
// tree-shakeable-adjacent — the DOM half never runs otherwise. Styling uses
// DESIGN.md theme tokens so it looks like voltradeai.com and not a library
// demo, and it pins to a corner clear of the map controls (Standing UI Law:
// it may not cover another element).
//
// The pure half — `hudRows`, `sparkline`, `perfEnabled` — is unit-tested;
// the DOM half is a thin writer over it.

import { FRAME, PRIORITY, type FrameLoop } from "./frameCore.ts";
import { METRIC, formatBytes, getCounter, getGauge, snapshot } from "./perfMetrics.ts";

/** How often the HUD rewrites its DOM. The loop runs at 60Hz; repainting
 *  text that fast is itself a frame cost and makes the numbers unreadable. */
export const HUD_REPAINT_MS = 250;

/** Sparkline width in samples. */
export const HUD_SPARK_SAMPLES = 48;

// ── enablement (pure) ────────────────────────────────────────────────────────

/**
 * Is the HUD requested? `?perf=1` in the query string, or a `vt:perf`
 * localStorage key so it survives navigation while debugging.
 * Accepts explicit inputs so it is testable without a DOM.
 */
export function perfEnabled(search = "", stored: string | null = null): boolean {
  if (stored === "1" || stored === "true") return true;
  if (!search) return false;
  const q = search.startsWith("?") ? search.slice(1) : search;
  for (const part of q.split("&")) {
    const [k, v = ""] = part.split("=");
    if (k === "perf") return v === "" || v === "1" || v === "true";
  }
  return false;
}

/** Read enablement from the live host, if there is one. */
export function perfEnabledFromHost(): boolean {
  const g = globalThis as unknown as {
    location?: { search?: string };
    localStorage?: { getItem(k: string): string | null };
  };
  let stored: string | null = null;
  try {
    stored = g.localStorage?.getItem("vt:perf") ?? null;
  } catch {
    // Private-mode / blocked storage. Query string still works.
  }
  return perfEnabled(g.location?.search ?? "", stored);
}

// ── rows (pure) ──────────────────────────────────────────────────────────────

export interface HudRow {
  label: string;
  value: string;
  /** "ok" | "warn" | "bad" — drives the value colour. */
  level: "ok" | "warn" | "bad";
}

export interface HudInput {
  p50: number;
  p95: number;
  worst: number;
  frames: number;
  samples: number;
  overBudget: number;
  errorCount: number;
  awakeSprings: number;
  registrations: number;
  paused: boolean;
}

/**
 * The HUD's rows. Kept pure so the thresholds — the ones the acceptance
 * criteria are written against — are asserted in a unit test rather than
 * eyeballed on a phone.
 */
export function hudRows(input: HudInput): HudRow[] {
  const budget = FRAME.FRAME_BUDGET_MS;
  const frameLevel = (ms: number): HudRow["level"] => (ms <= budget ? "ok" : ms <= budget * 2 ? "warn" : "bad");

  const vram = getGauge(METRIC.VRAM_BYTES);
  const inFlight = getGauge(METRIC.IN_FLIGHT);
  const unready = getCounter(METRIC.UNREADY_DRAWS);

  const rows: HudRow[] = [
    { label: "frame p50", value: `${input.p50.toFixed(1)} ms`, level: frameLevel(input.p50) },
    { label: "frame p95", value: `${input.p95.toFixed(1)} ms`, level: frameLevel(input.p95) },
    { label: "worst", value: `${input.worst.toFixed(1)} ms`, level: frameLevel(input.worst) },
    {
      label: "over budget",
      value: `${input.overBudget}/${input.samples}`,
      level: input.overBudget === 0 ? "ok" : input.overBudget * 20 > input.samples ? "bad" : "warn",
    },
    { label: "draw calls", value: String(getGauge(METRIC.DRAW_CALLS)), level: "ok" },
    { label: "textures", value: String(getGauge(METRIC.TEXTURES)), level: "ok" },
    { label: "vram est", value: formatBytes(vram), level: "ok" },
    {
      label: "in flight",
      value: String(inFlight),
      level: inFlight > 16 ? "warn" : "ok",
    },
    {
      label: "tiles",
      value: `${getGauge(METRIC.TILES_RESIDENT)} res / ${getGauge(METRIC.TILES_PENDING)} pend`,
      level: "ok",
    },
    { label: "features", value: String(getGauge(METRIC.FEATURES)), level: "ok" },
    // Law II.1 — an unready draw is a bug, not a metric with a healthy
    // nonzero range. Any value above zero is red.
    { label: "unready draws", value: String(unready), level: unready === 0 ? "ok" : "bad" },
    {
      label: "springs",
      value: `${input.awakeSprings} awake`,
      level: "ok",
    },
    {
      label: "callbacks",
      value: `${input.registrations}${input.paused ? " (paused)" : ""}`,
      level: input.paused ? "warn" : "ok",
    },
  ];
  if (input.errorCount > 0) {
    rows.push({ label: "cb errors", value: String(input.errorCount), level: "bad" });
  }
  return rows;
}

/** A tiny ASCII sparkline of recent frame times, so a periodic hitch is
 *  visible as a shape rather than inferred from a moving p95. */
export function sparkline(samples: readonly number[], maxMs = FRAME.FRAME_BUDGET_MS * 3): string {
  if (!samples.length) return "";
  const glyphs = "▁▂▃▄▅▆▇█";
  const take = samples.slice(-HUD_SPARK_SAMPLES);
  let out = "";
  for (const s of take) {
    const t = Math.max(0, Math.min(1, s / maxMs));
    out += glyphs[Math.min(glyphs.length - 1, Math.round(t * (glyphs.length - 1)))];
  }
  return out;
}

// ── the overlay (DOM) ────────────────────────────────────────────────────────

export interface PerfHudHandle {
  /** Remove the overlay and unregister from the loop (Law IV teardown). */
  dispose(): void;
  /** The overlay element, or null when the HUD is disabled. */
  element: HTMLElement | null;
}

const HUD_STYLE = [
  "position:fixed",
  "right:8px",
  // Clear of the map's bottom-right controls and of the phone's home bar.
  "bottom:calc(8px + env(safe-area-inset-bottom, 0px))",
  "z-index:2147483000",
  "min-width:186px",
  "max-width:min(52vw, 260px)",
  "padding:8px 10px",
  "border-radius:8px",
  "background:var(--bg-card, rgba(15,29,51,0.6))",
  "backdrop-filter:blur(8px)",
  "-webkit-backdrop-filter:blur(8px)",
  "border:1px solid var(--border, rgba(120,165,220,0.16))",
  "color:var(--text-secondary, #b3c2d8)",
  "font:11px/1.45 ui-monospace, SFMono-Regular, Menlo, monospace",
  "pointer-events:none",
  "user-select:none",
  "white-space:nowrap",
].join(";");

const LEVEL_COLOR: Record<HudRow["level"], string> = {
  ok: "var(--text-primary, #eef3fb)",
  warn: "var(--accent-orange, #fbb24c)",
  bad: "var(--accent-red, #ff5a6e)",
};

/**
 * Mount the HUD against a loop. Returns a handle whose `dispose()` fully
 * tears down — element removed, callback unregistered, no retained closure.
 * A no-op handle is returned when the HUD is not requested or there is no
 * DOM, so callers can mount unconditionally.
 */
export function mountPerfHud(loop: FrameLoop, opts: { enabled?: boolean; parent?: HTMLElement } = {}): PerfHudHandle {
  const enabled = opts.enabled ?? perfEnabledFromHost();
  const doc = (globalThis as unknown as { document?: Document }).document;
  if (!enabled || !doc || typeof doc.createElement !== "function") {
    return { dispose: () => {}, element: null };
  }

  const el = doc.createElement("div");
  el.setAttribute("data-vt-perf-hud", "1");
  el.setAttribute("aria-hidden", "true");
  el.style.cssText = HUD_STYLE;
  (opts.parent ?? doc.body).appendChild(el);

  const recent: number[] = [];
  let lastPaint = -Infinity;
  let lastFrames = 0;
  let lastPaintNow = 0;

  const unregister = loop.register(
    (_dt, now) => {
      recent.push(loop.dt);
      if (recent.length > HUD_SPARK_SAMPLES * 2) recent.splice(0, recent.length - HUD_SPARK_SAMPLES * 2);
      if (now - lastPaint < HUD_REPAINT_MS) return;
      const elapsed = now - lastPaintNow;
      const fps = elapsed > 0 ? ((loop.frameCount - lastFrames) * 1000) / elapsed : 0;
      lastPaint = now;
      lastPaintNow = now;
      lastFrames = loop.frameCount;

      const st = loop.stats();
      const rows = hudRows({
        p50: st.p50,
        p95: st.p95,
        worst: st.worst,
        frames: st.frames,
        samples: st.samples,
        overBudget: st.overBudget,
        errorCount: loop.errorCount,
        awakeSprings: loop.awakeSpringCount,
        registrations: loop.registrationCount,
        paused: loop.isPaused,
      });

      const head =
        `<div style="color:var(--accent,#4d9fff);letter-spacing:.06em;font-size:10px;margin-bottom:4px">` +
        `PERF · ${fps.toFixed(0)} fps</div>`;
      const spark =
        `<div style="color:var(--text-tertiary,#6680a0);margin-bottom:4px;overflow:hidden">` +
        `${sparkline(recent)}</div>`;
      const body = rows
        .map(
          (r) =>
            `<div style="display:flex;justify-content:space-between;gap:10px">` +
            `<span style="color:var(--text-tertiary,#6680a0)">${r.label}</span>` +
            `<span style="color:${LEVEL_COLOR[r.level]}">${r.value}</span></div>`,
        )
        .join("");
      el.innerHTML = head + spark + body;
    },
    PRIORITY.RENDER,
    { label: "perfHud" },
  );

  let disposed = false;
  return {
    element: el,
    dispose() {
      if (disposed) return;
      disposed = true;
      unregister();
      el.remove();
      recent.length = 0;
    },
  };
}

/** Re-exported so consumers need one import for "is perf on". */
export { snapshot as perfSnapshot };
