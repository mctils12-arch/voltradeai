// PANEL LAYOUT — movable / minimizable / lockable floating panels with
// automatically remembered placement (human directive 2026-07-20: "you
// need to be able to move them, minimize them, all controls, and it to
// remember where you put them automatically … and the ability to lock
// them in place so you dont accidently move them").
//
// One localStorage record holds every panel's preferences:
//   { [panelId]: { pos?: {left, top}, min?: boolean, locked?: boolean } }
//
// Coordinates are CONTAINER-relative (the panel's offsetParent — the map
// page root), not viewport-relative: the topbar and side rail offset the
// container, and container coords survive chrome changes. On restore the
// saved position is clamped so a panel saved on a big window can never
// end up unreachable on a small one (the grip always stays on screen).
//
// Phones (<768px) are exempt: cards are bottom sheets there (DESIGN.md),
// custom placement would fight the sheet layout — handlers no-op and the
// grips/locks hide via CSS.
//
// Pure core (parse/clamp/merge) is unit-tested with `npx tsx --test`;
// the storage + DOM wrappers are thin and no-throw.

export interface PanelPos { left: number; top: number }
export interface PanelPrefs { pos?: PanelPos; min?: boolean; locked?: boolean }
export type PanelLayout = Record<string, PanelPrefs>;

export const PANEL_LAYOUT_KEY = "vt-panel-layout-v1";
/** below this viewport width custom placement is off (bottom-sheet land). */
export const PANEL_DRAG_MIN_W = 768;
/** the panel's top-left must stay at least this far inside the container
 *  (grip reachable), matching the pre-existing card drag rule. */
export const PANEL_KEEP_X = 80;
export const PANEL_KEEP_Y = 60;

/** Tolerant parse — bad JSON, wrong shapes, or non-finite numbers degrade
 *  to "no preference", never a throw (a corrupt record must not brick the
 *  page). Pure. */
export function parseLayout(raw: string | null | undefined): PanelLayout {
  if (!raw) return {};
  try {
    const obj = JSON.parse(raw);
    if (!obj || typeof obj !== "object" || Array.isArray(obj)) return {};
    const out: PanelLayout = {};
    for (const [id, v] of Object.entries(obj as Record<string, unknown>)) {
      if (!v || typeof v !== "object") continue;
      const p = v as Record<string, unknown>;
      const prefs: PanelPrefs = {};
      const pos = p.pos as Record<string, unknown> | undefined;
      if (pos && typeof pos === "object" &&
          Number.isFinite(pos.left) && Number.isFinite(pos.top)) {
        prefs.pos = { left: pos.left as number, top: pos.top as number };
      }
      if (typeof p.min === "boolean") prefs.min = p.min;
      if (typeof p.locked === "boolean") prefs.locked = p.locked;
      out[id] = prefs;
    }
    return out;
  } catch {
    return {};
  }
}

/** Clamp a container-relative position so the panel's grip corner stays
 *  reachable inside a container of the given size. Pure. */
export function clampPos(pos: PanelPos, cw: number, ch: number): PanelPos {
  return {
    left: Math.max(0, Math.min(Math.max(0, cw - PANEL_KEEP_X), pos.left)),
    top: Math.max(0, Math.min(Math.max(0, ch - PANEL_KEEP_Y), pos.top)),
  };
}

/** Merge a preference patch into a layout (pos/min/locked update
 *  independently; explicit `undefined` in the patch deletes the field —
 *  how double-click-reset forgets a position). Pure. */
export function mergePrefs(layout: PanelLayout, id: string, patch: Partial<PanelPrefs>): PanelLayout {
  const cur = { ...(layout[id] ?? {}) };
  for (const k of ["pos", "min", "locked"] as const) {
    if (k in patch) {
      const v = patch[k];
      if (v === undefined) delete cur[k];
      else (cur as Record<string, unknown>)[k] = v;
    }
  }
  return { ...layout, [id]: cur };
}

// ── storage wrappers (no-throw; privacy modes without storage degrade to
//    session-only behavior) ────────────────────────────────────────────────
export function loadLayout(): PanelLayout {
  try {
    return parseLayout(window.localStorage.getItem(PANEL_LAYOUT_KEY));
  } catch {
    return {};
  }
}

export function getPanelPrefs(id: string): PanelPrefs {
  return loadLayout()[id] ?? {};
}

export function savePanelPrefs(id: string, patch: Partial<PanelPrefs>): void {
  try {
    const next = mergePrefs(loadLayout(), id, patch);
    window.localStorage.setItem(PANEL_LAYOUT_KEY, JSON.stringify(next));
  } catch { /* storage unavailable — live with defaults */ }
}

// ── DOM wrappers ────────────────────────────────────────────────────────────
const containerOf = (el: HTMLElement): HTMLElement =>
  (el.offsetParent as HTMLElement | null) ?? document.body;

/** Apply a panel's saved position (clamped to the CURRENT container size).
 *  Returns false when there is no saved position (CSS default applies). */
/** Un-anchoring a panel that gets its width from BOTH edges (left+right
 *  CSS, e.g. the flight profile bar) would collapse it — freeze the
 *  current rendered width before switching to left/top placement. */
const freezeWidth = (el: HTMLElement) => {
  if (!el.style.width) el.style.width = `${el.getBoundingClientRect().width}px`;
};

export function applyPanelPos(el: HTMLElement | null, id: string): boolean {
  if (!el) return false;
  if (typeof window !== "undefined" && window.innerWidth < PANEL_DRAG_MIN_W) return false;
  const prefs = getPanelPrefs(id);
  if (!prefs.pos) return false;
  const box = containerOf(el).getBoundingClientRect();
  const p = clampPos(prefs.pos, box.width, box.height);
  freezeWidth(el);
  el.style.left = `${p.left}px`;
  el.style.top = `${p.top}px`;
  el.style.right = "auto";
  el.style.bottom = "auto";
  return true;
}

/** Back to the stylesheet's default anchor (reset / no saved position). */
export function clearPanelPos(el: HTMLElement | null): void {
  if (!el) return;
  el.style.left = "";
  el.style.top = "";
  el.style.right = "";
  el.style.bottom = "";
  el.style.width = "";
}

/** structural pointer-event shape — works for React synthetic and native */
interface PtrEv {
  clientX: number;
  clientY: number;
  pointerId: number;
  target: EventTarget | null;
  preventDefault(): void;
}

export interface PanelDragProps {
  onPointerDown(e: PtrEv): void;
  onPointerMove(e: PtrEv): void;
  onPointerUp(e: PtrEv): void;
  onPointerCancel(e: PtrEv): void;
  onDoubleClick(e: { target: EventTarget | null }): void;
}

/**
 * Drag-to-move handlers for a panel grip. Create ONCE per mounted panel
 * (useMemo/useRef) — the closure carries the drag state. Presses on
 * buttons inside the grip pass through untouched; a locked panel ignores
 * drags entirely; double-click on the grip forgets the saved position
 * (back to the default spot). Positions save on release, automatically.
 */
export function panelDragProps(
  id: string,
  getEl: () => HTMLElement | null,
  isLocked: () => boolean,
  onMoved?: () => void,
): PanelDragProps {
  let drag: { dx: number; dy: number; moved: boolean } | null = null;
  const release = () => {
    const el = getEl();
    if (drag?.moved && el) {
      const box = containerOf(el).getBoundingClientRect();
      const r = el.getBoundingClientRect();
      savePanelPrefs(id, { pos: { left: r.left - box.left, top: r.top - box.top } });
      onMoved?.();
    }
    drag = null;
  };
  return {
    onPointerDown(e) {
      if (typeof window !== "undefined" && window.innerWidth < PANEL_DRAG_MIN_W) return;
      if (isLocked()) return;
      if ((e.target as Element | null)?.closest?.("button")) return; // buttons stay buttons
      const el = getEl();
      if (!el) return;
      const r = el.getBoundingClientRect();
      drag = { dx: e.clientX - r.left, dy: e.clientY - r.top, moved: false };
      (e.target as Element | null)?.setPointerCapture?.(e.pointerId);
      e.preventDefault();
    },
    onPointerMove(e) {
      const el = getEl();
      if (!el || !drag) return;
      if (!drag.moved) freezeWidth(el); // before the anchors change
      const box = containerOf(el).getBoundingClientRect();
      const p = clampPos(
        { left: e.clientX - drag.dx - box.left, top: e.clientY - drag.dy - box.top },
        box.width, box.height,
      );
      drag.moved = true;
      el.style.left = `${p.left}px`;
      el.style.top = `${p.top}px`;
      el.style.right = "auto";
      el.style.bottom = "auto";
    },
    onPointerUp: release,
    onPointerCancel: release,
    onDoubleClick(e) {
      if ((e.target as Element | null)?.closest?.("button")) return;
      savePanelPrefs(id, { pos: undefined });
      clearPanelPos(getEl());
    },
  };
}
