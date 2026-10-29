// Cockpit layout engine — the hook a cockpit page runs its panels on.
// 2026-10-29: lifted out of client/src/bot/BotPage.tsx so the Trade page runs
// the SAME engine, not a copy. It owns the flags (hold / held / pinned /
// floating / zoomed), the float-window rects, the pop-out windows and the
// BroadcastChannel they share with the page, the slot elements (useFlip
// measures them), the hold snapshots, the handlers the panel chrome calls and
// the Esc-docks-everything listener. What a panel SHOWS stays the page's: it
// builds the bodies, calls `heldBodies` once per render and `chrome(id, now)`
// per panel. The flag transitions are pure functions (tested headless; the
// hook applies them); no DOM is touched at import time.
import { useCallback, useEffect, useRef, useState, type Dispatch, type MutableRefObject, type ReactNode, type RefObject, type SetStateAction } from "react";
import { emptyFlags, initialFloatRect, isPinned, toggleIn, type LayoutFlags, type Rect, type Tier } from "./zoneModel";
import { paneFeatures, paneLifecycleMsg, paneName, paneUrl, parseChannelMsg, windowId, type ChannelMsg, type SlotRect } from "./popout";
// the engine's companion hooks, reachable from this module too: a cockpit page
// imports its engine from one place (the Trade page does), index.ts is the barrel
export { useTier, reducedMotion, type Viewport } from "./useTier";
export { useNow } from "./useNow";
export { useFlip, type FlipOptions } from "./useFlip";

/** this window on the pane channel: a sender ignores its own echoes */
export const WIN_ID = windowId();
/** defaults a cockpit overrides so its windows stay apart from another cockpit's */
export const DEFAULT_CHANNEL = "vt-cockpit-panes";
export const DEFAULT_HASH_PREFIX = "#/pane/";
export const DEFAULT_WINDOW_PREFIX = "vt-pane";

// ── flag transitions (pure) ─────────────────────────────────────────────────
export function applyHold<Id extends string>(f: LayoutFlags<Id>, id: Id): LayoutFlags<Id> { return { ...f, held: toggleIn(f.held, id) }; }
export function applyPin<Id extends string>(f: LayoutFlags<Id>, id: Id): LayoutFlags<Id> { return { ...f, pinned: toggleIn(f.pinned, id) }; }
/** the page-wide hold (the KPI strip's control): the director pauses, every feed stays live */
export function applyHoldAll<Id extends string>(f: LayoutFlags<Id>): LayoutFlags<Id> { return { ...f, hold: !f.hold }; }
/** the same object back when already floating: the caller skips its rect work */
export function applyFloat<Id extends string>(f: LayoutFlags<Id>, id: Id): LayoutFlags<Id> { return f.floating.includes(id) ? f : { ...f, floating: [...f.floating, id] }; }
/** the window docks; the zoom ends only when it was this panel's */
export function applyDock<Id extends string>(f: LayoutFlags<Id>, id: Id): LayoutFlags<Id> { return { ...f, floating: f.floating.filter((x) => x !== id), zoomed: f.zoomed === id ? null : f.zoomed }; }
/** zoom toggles; a zoomed panel is not also a window */
export function applyZoom<Id extends string>(f: LayoutFlags<Id>, id: Id): LayoutFlags<Id> { return { ...f, zoomed: f.zoomed === id ? null : id, floating: f.floating.filter((x) => x !== id) }; }
/** its own browser window: the panel leaves the page's windows and the zoom (the ghost slot stays) */
export function applyPop<Id extends string>(f: LayoutFlags<Id>, id: Id): LayoutFlags<Id> { return applyDock(f, id); }
/** Esc: every window docks, the zoom ends; held and pinned stay */
export function applyDockAll<Id extends string>(f: LayoutFlags<Id>): LayoutFlags<Id> { return { ...f, zoomed: null, floating: [] }; }

// ── hold snapshots ──────────────────────────────────────────────────────────
export interface HeldSnapshot { since: number; body: ReactNode; ticks: number }

/** One render's hold ledger: a newly held panel is snapshotted now (its body as
 *  built this render, the tick count), a still-held one keeps its snapshot, a
 *  released one is dropped. Runs before the chrome reads it, so the first held
 *  frame already says since-when and +0. */
export function syncHeld<Id extends string>(held: Map<Id, HeldSnapshot>, heldIds: readonly Id[], fresh: (id: Id) => ReactNode, nowMs: number, tickCount: number): void {
  for (const id of heldIds) if (!held.has(id)) held.set(id, { since: nowMs, body: fresh(id), ticks: tickCount });
  for (const id of [...held.keys()]) if (!heldIds.includes(id)) held.delete(id);
}

/** the pop-out control is offered: not on touch, not on the phone tier */
export function canPopOut(touch: boolean, tier: Tier): boolean { return !touch && tier !== "s"; }

/** a DOMRect's four numbers (what the geometry here reads) */
export interface ClientBox { left: number; top: number; width: number; height: number }

/** a slot's box in page coordinates (the page root's offset removed); a slot
 *  not in the DOM gets a default so the window still opens */
export function pageSlot(r: ClientBox | null, root: { left: number; top: number } | undefined): Rect {
  return r ? { x: r.left - (root?.left ?? 0), y: r.top - (root?.top ?? 0), w: r.width, h: r.height } : { x: 100, y: 100, w: 600, h: 400 };
}

/** the slot in screen coordinates (the popup opens over it); the browser
 *  chrome above the viewport is outerHeight − innerHeight */
export function screenSlot(r: ClientBox, win: { screenX: number; screenY: number; outerHeight: number; innerHeight: number }): SlotRect {
  return { x: r.left + win.screenX, y: r.top + win.screenY + Math.max(0, win.outerHeight - win.innerHeight), w: r.width, h: r.height };
}

// ── the chrome a panel header renders from ──────────────────────────────────
export interface Chrome<Id extends string> {
  id: Id; held: boolean; pinned: boolean; heldSince: number | null; ticksSinceHold: number; nowMs: number;
  onHold: (id: Id) => void; onPin: (id: Id) => void; onFloat: (id: Id) => void;
  /** open this panel in its own browser window (popout.ts) */
  onPop: (id: Id) => void;
  /** this panel IS its own browser window: nothing to pin to or float over */
  pane: boolean;
  /** the pop-out control is offered (not on touch / the phone tier) */
  canPop: boolean;
}

export interface CockpitLayoutInput<Id extends string> {
  /** every panel id the page can show (a pane message must name one of them) */
  ids: readonly Id[];
  /** the panel titles: the pane's tab title, the pop-up-blocked note */
  titles: Record<Id, string>;
  tier: Tier;
  touch: boolean;
  /** the page's size — the float rects' bounds when the root is not measurable */
  page: { w: number; h: number };
  /** the page root; float rects are relative to it */
  rootRef: RefObject<HTMLElement>;
  /** this window IS one panel popped out of the page (null on the page itself) */
  paneId: Id | null;
  /** data ticks so far: a held panel shows how many it has missed */
  tickCount: number;
  /** a pane's tab title is `${titles[paneId]}${paneTitleSuffix}` */
  paneTitleSuffix?: string;
  /** the BroadcastChannel this page shares with its panes */
  channelName?: string;
  /** this window's id on that channel (default: one per page load) */
  windowId?: string;
  /** the pop-out URL is `${origin}${pathname}${paneHashPrefix}${id}` — the page's router must render that hash as the one panel */
  paneHashPrefix?: string;
  /** one named browser window per panel: `${paneWindowPrefix}-${id}` */
  paneWindowPrefix?: string;
  /** a channel message that is not a pane lifecycle (the bot's focus symbol): the page's to interpret; echoes of this window are already dropped */
  onMessage?: (m: ChannelMsg) => void;
  /** whether the chrome offers the pop-out control; default: not on touch, not on the phone tier (canPopOut) — a page with no pane route passes false */
  canPop?: boolean;
}

export interface CockpitLayout<Id extends string> {
  flags: LayoutFlags<Id>;
  setFlags: Dispatch<SetStateAction<LayoutFlags<Id>>>;
  /** float-window rects, page coordinates */
  rects: Partial<Record<Id, Rect>>;
  setRects: Dispatch<SetStateAction<Partial<Record<Id, Rect>>>>;
  setRect: (id: Id, r: Rect) => void;
  /** the panels in their own browser windows (their slots show a ghost) */
  popped: Id[];
  /** the pop-up-blocked note, or null */
  popNote: string | null;
  setPopNote: Dispatch<SetStateAction<string | null>>;
  /** the slot elements by id (useFlip measures them; float and pop-out read their boxes) */
  slots: MutableRefObject<Map<Id, HTMLElement>>;
  slotRef: (id: Id) => (el: HTMLElement | null) => void;
  held: MutableRefObject<Map<Id, HeldSnapshot>>;
  /** once per render, after the fresh bodies are built: snapshots the newly held (at `nowMs`, the page's clock — the held-since line reads it), drops the released, returns the body accessor (the snapshot for a held panel, the fresh one otherwise) */
  heldBodies: (fresh: Record<Id, ReactNode>, nowMs?: number) => (id: Id) => ReactNode;
  onHold: (id: Id) => void;
  onPin: (id: Id) => void;
  onHoldAll: () => void;
  onFloat: (id: Id) => void;
  onDock: (id: Id) => void;
  onZoom: (id: Id) => void;
  onPop: (id: Id) => void;
  /** bring a popped panel's window forward */
  onRaise: (id: Id) => void;
  /** close a popped panel's window; its slot refills */
  onBringBack: (id: Id) => void;
  /** a window or a zoom is open (Esc is listened for) */
  windowsOpen: boolean;
  chrome: (id: Id, nowMs: number) => Chrome<Id>;
  /** send to every other window of this page; `from` is filled in */
  post: (msg: { t: string } & Record<string, unknown>) => void;
  windowId: string;
}

export function useCockpitLayout<Id extends string>(input: CockpitLayoutInput<Id>): CockpitLayout<Id> {
  const { ids, titles, tier, touch, page, rootRef, paneId, tickCount, paneTitleSuffix = "", channelName = DEFAULT_CHANNEL, windowId: winId = WIN_ID, paneHashPrefix = DEFAULT_HASH_PREFIX, paneWindowPrefix = DEFAULT_WINDOW_PREFIX, onMessage } = input;
  const canPop = input.canPop ?? canPopOut(touch, tier);
  const pageW = page.w, pageH = page.h;
  const [flags, setFlags] = useState<LayoutFlags<Id>>(() => emptyFlags<Id>());
  const [rects, setRects] = useState<Partial<Record<Id, Rect>>>({});
  // pop-out windows (popout.ts): `popped` the panels this page has sent to
  // their own windows (their handles in popWins; the slot shows a ghost until
  // they come back)
  const [popped, setPopped] = useState<Id[]>([]);
  const popWins = useRef<Map<Id, Window>>(new Map());
  const [popNote, setPopNote] = useState<string | null>(null);
  const chan = useRef<BroadcastChannel | null>(null);
  const slots = useRef<Map<Id, HTMLElement>>(new Map());
  const held = useRef<Map<Id, HeldSnapshot>>(new Map());
  // read through refs by the channel handler, so a page passing a fresh
  // callback or array each render does not re-open the channel
  const onMsg = useRef(onMessage); onMsg.current = onMessage;
  const idsRef = useRef(ids); idsRef.current = ids;

  const onHold = useCallback((id: Id) => setFlags((f) => applyHold(f, id)), []);
  const onPin = useCallback((id: Id) => setFlags((f) => applyPin(f, id)), []);
  const onHoldAll = useCallback(() => setFlags((f) => applyHoldAll(f)), []);
  const setRect = useCallback((id: Id, r: Rect) => setRects((rs) => { const next = { ...rs }; next[id] = r; return next; }), []);
  const onFloat = useCallback((id: Id) => setFlags((f) => {
    if (f.floating.includes(id)) return f;
    const el = slots.current.get(id); const r = el ? el.getBoundingClientRect() : null;
    const root = rootRef.current?.getBoundingClientRect();
    const rect = initialFloatRect(pageSlot(r, root), { w: root?.width ?? pageW, h: root?.height ?? pageH });
    setRects((rs) => { const next = { ...rs }; next[id] = rect; return next; });
    return applyFloat(f, id);
  }), [pageW, pageH, rootRef]);
  const onDock = useCallback((id: Id) => setFlags((f) => applyDock(f, id)), []);
  const onZoom = useCallback((id: Id) => setFlags((f) => applyZoom(f, id)), []);
  const post = useCallback((msg: { t: string } & Record<string, unknown>) => { chan.current?.postMessage({ ...msg, from: winId }); }, [winId]);
  useEffect(() => {
    if (typeof BroadcastChannel === "undefined") return;
    const ch = new BroadcastChannel(channelName);
    chan.current = ch;
    ch.onmessage = (e: MessageEvent) => {
      const m = parseChannelMsg(e.data);
      if (!m || m.from === winId) return;
      const life = paneLifecycleMsg(m, idsRef.current);
      if (life) {
        if (life.t === "pane-closed") { popWins.current.delete(life.id); setPopped((p) => p.filter((x) => x !== life.id)); }
        return;
      }
      onMsg.current?.(m);
    };
    return () => { ch.close(); chan.current = null; };
  }, [channelName, winId]);
  // a pane names itself in the tab title and says goodbye when it closes
  useEffect(() => {
    if (paneId === null) return;
    const prev = document.title;
    document.title = `${titles[paneId]}${paneTitleSuffix}`;
    const bye = () => chan.current?.postMessage({ t: "pane-closed", id: paneId, from: winId });
    window.addEventListener("pagehide", bye);
    return () => { window.removeEventListener("pagehide", bye); document.title = prev; };
  }, [paneId, titles, paneTitleSuffix, winId]);
  const onPop = useCallback((id: Id) => {
    const el = slots.current.get(id); const r = el ? el.getBoundingClientRect() : null;
    // the slot in screen coordinates (the popup opens over it)
    const slot = r ? screenSlot(r, window) : null;
    const w = window.open(paneUrl(id, paneHashPrefix, window.location), paneName(id, paneWindowPrefix), paneFeatures(slot, { w: window.screen.availWidth, h: window.screen.availHeight }));
    if (!w) { setPopNote(`${titles[id]}: the browser blocked the pop-up window — allow pop-ups for this site, then try again`); return; }
    setPopNote(null);
    popWins.current.set(id, w);
    w.focus();
    setFlags((f) => applyPop(f, id));
    setPopped((p) => (p.includes(id) ? p : [...p, id]));
  }, [titles, paneHashPrefix, paneWindowPrefix]);
  const onRaise = useCallback((id: Id) => { const w = popWins.current.get(id); if (w && !w.closed) w.focus(); }, []);
  const onBringBack = useCallback((id: Id) => {
    const w = popWins.current.get(id);
    if (w && !w.closed) w.close();
    popWins.current.delete(id);
    setPopped((p) => p.filter((x) => x !== id));
  }, []);
  // a pane closed from its own window: its goodbye on the channel is the
  // fast path; this poll catches a window killed without a pagehide
  useEffect(() => {
    if (!popped.length) return;
    const t = window.setInterval(() => {
      for (const [id, w] of popWins.current) if (w.closed) { popWins.current.delete(id); setPopped((p) => p.filter((x) => x !== id)); }
    }, 1000);
    return () => window.clearInterval(t);
  }, [popped.length]);
  // Esc docks every window and leaves the zoom. Listened on the document
  // while one is open: the page root only received keys while focus was
  // inside it, and a click on the page body left focus outside (2026-10-20).
  const windowsOpen = flags.floating.length > 0 || flags.zoomed !== null;
  useEffect(() => {
    if (!windowsOpen) return;
    const onEsc = (e: KeyboardEvent) => { if (e.key === "Escape") setFlags((f) => applyDockAll(f)); };
    document.addEventListener("keydown", onEsc);
    return () => document.removeEventListener("keydown", onEsc);
  }, [windowsOpen]);

  const slotRef = useCallback((id: Id) => (el: HTMLElement | null) => { if (el) slots.current.set(id, el); else slots.current.delete(id); }, []);
  const heldBodies = useCallback((fresh: Record<Id, ReactNode>, nowMs: number = Date.now()) => {
    syncHeld(held.current, flags.held, (id) => fresh[id], nowMs, tickCount);
    return (id: Id): ReactNode => held.current.get(id)?.body ?? fresh[id];
  }, [flags.held, tickCount]);
  const chrome = useCallback((id: Id, nowMs: number): Chrome<Id> => {
    const h = held.current.get(id);
    return { id, held: flags.held.includes(id), pinned: isPinned(flags, id), heldSince: h?.since ?? null, ticksSinceHold: h ? tickCount - h.ticks : 0, nowMs, onHold, onPin, onFloat, onPop, pane: paneId !== null, canPop };
  }, [flags, tickCount, onHold, onPin, onFloat, onPop, paneId, canPop]);

  return { flags, setFlags, rects, setRects, setRect, popped, popNote, setPopNote, slots, slotRef, held, heldBodies, onHold, onPin, onHoldAll, onFloat, onDock, onZoom, onPop, onRaise, onBringBack, windowsOpen, chrome, post, windowId: winId };
}
