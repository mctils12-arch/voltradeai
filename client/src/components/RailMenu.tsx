// RAIL MENU — the one way a button in the /data map's top-left icon rail
// opens a menu. Owner report 2026-11-19: "when you click on the mountain, it
// covers the other icon and moves down. Why does it move? It's where I have
// my mouse." The base-map-style menu used to be laid out in flow beside its
// button, so opening it re-centred the button 23px lower, onto the button
// below. Any future rail menu goes through this component instead of
// re-deriving that layout:
//
//   • the button's slot is a FIXED 44px box — the menu is positioned out of
//     flow (lib/railFlyout), so opening it cannot move or cover a rail button;
//   • the menu opens to the right of the rail, stays inside the map page at
//     every width, and steps aside for any control already there (Standing UI
//     Law: no popup covers another element; never off-page);
//   • one click (or tap) opens, a second click on the same button closes;
//     Escape and a press anywhere outside close it too;
//   • a mouse still opens it on hover and it still goes away when the mouse
//     leaves (human 2026-07-21) — but the click that follows a hover-open
//     KEEPS it open rather than toggling it shut under the pointer, and touch
//     never takes the hover path (a tap used to open-then-close in one go).
//
// Controlled: the page owns `open` (it also closes the menu when an option is
// picked). Positioning is written straight to the menu element before paint,
// on open and whenever the page or the menu changes size — never from a map
// event (Law I).

import { useEffect, useId, useLayoutEffect, useRef, type PointerEvent as ReactPointerEvent, type ReactNode } from "react";
import { isBackdrop, placeRailFlyout, type Box } from "@/lib/railFlyout";

/** a mouse that leaves the menu has this long to come back before it closes
 *  (crossing the 6px gap between button and menu is a leave) */
const LEAVE_GRACE_MS = 250;
/** Boxes (page coordinates → container coordinates) of everything on the map
 *  page a menu must not cover: every visible child except the menu's own slot
 *  and backdrops. A pointer-events:none wrapper is a layout shell here (nav
 *  cluster, controls column) — its children are the controls. */
function collectObstacles(container: Element, self: Element): Box[] {
  const c = container.getBoundingClientRect();
  const out: Box[] = [];
  const walk = (parent: Element, depth: number) => {
    for (const el of Array.from(parent.children)) {
      if (el === self || el.contains(self)) continue;
      const cs = getComputedStyle(el);
      if (cs.display === "none" || cs.visibility === "hidden" || Number(cs.opacity) === 0) continue;
      if (cs.pointerEvents === "none" && el.children.length > 0 && depth < 3) { walk(el, depth + 1); continue; }
      const r = el.getBoundingClientRect();
      if (r.width < 1 || r.height < 1) continue;
      if (isBackdrop(r, c)) continue;
      out.push({ left: r.left - c.left, top: r.top - c.top, width: r.width, height: r.height });
    }
  };
  walk(container, 0);
  return out;
}

export interface RailMenuProps {
  /** stable name: data-vt-rail-menu / data-vt-rail-flyout carry it */
  id: string;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  /** places the 44px slot in the rail (its `top`) */
  className?: string;
  /** added to the slot while the menu is open */
  openClassName?: string;
  /** aria-label of the button + menu group */
  groupLabel: string;
  buttonClassName?: string;
  buttonLabel: string;
  buttonTitle: string;
  icon: ReactNode;
  menuClassName?: string;
  children: ReactNode;
}

export function RailMenu(props: RailMenuProps) {
  const { id, open, onOpenChange } = props;
  const slotRef = useRef<HTMLDivElement | null>(null);
  const btnRef = useRef<HTMLButtonElement | null>(null);
  const menuRef = useRef<HTMLDivElement | null>(null);
  const menuId = useId();
  // true between a hover-open and the pointer leaving the button: the click
  // that lands in that window is the user "clicking the icon to open it"
  const hoverOpenedRef = useRef(false);
  const besideRef = useRef(true);
  const leaveTimerRef = useRef<number | null>(null);
  const changeRef = useRef(onOpenChange);
  changeRef.current = onOpenChange;

  const cancelLeave = () => {
    if (leaveTimerRef.current !== null) { window.clearTimeout(leaveTimerRef.current); leaveTimerRef.current = null; }
  };

  // placement: measured and written before paint; re-run when the page or
  // the menu's own content changes size
  useLayoutEffect(() => {
    if (!open) return;
    const slot = slotRef.current;
    const menu = menuRef.current;
    if (!slot || !menu) return;
    const container: Element = slot.offsetParent ?? document.documentElement;
    const place = () => {
      const c = container.getBoundingClientRect();
      const a = slot.getBoundingClientRect();
      const anchor: Box = { left: a.left - c.left, top: a.top - c.top, width: a.width, height: a.height };
      // the menu's size depends on how wide it may be (its rows wrap), so the
      // placement probes it at each width it considers
      menu.style.maxHeight = "none";
      const heights = new Map<number, number>();
      const p = placeRailFlyout({
        anchor,
        container: { width: c.width, height: c.height },
        measure: (maxWidth) => {
          menu.style.maxWidth = `${maxWidth}px`;
          // fractional box, rounded UP: offsetWidth/Height round to nearest,
          // which let a 152.3px menu be placed flush against a panel 152px
          // below and overlap it by a sliver
          const r = menu.getBoundingClientRect();
          heights.set(maxWidth, Math.ceil(r.height));
          return { width: Math.ceil(r.width), height: Math.ceil(r.height) };
        },
        obstacles: collectObstacles(container, slot),
      });
      menu.style.left = `${p.left - anchor.left}px`;
      menu.style.top = `${p.top - anchor.top}px`;
      menu.style.maxWidth = `${p.maxWidth}px`;
      menu.style.maxHeight = `${p.maxHeight}px`;
      menu.dataset.vtRailPlacement = p.mode;
      // scroll inside only when the page is shorter than the menu — an
      // always-on overflow would clip the pill rows' drop shadows
      menu.dataset.vtRailScroll = (heights.get(p.maxWidth) ?? 0) > p.maxHeight ? "true" : "false";
      besideRef.current = p.mode === "beside";
    };
    place();
    if (typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(place);
    ro.observe(container);
    ro.observe(menu);
    return () => ro.disconnect();
  }, [open]);

  // Escape and a press anywhere outside close the menu
  useEffect(() => {
    if (!open) return;
    const onDown = (e: PointerEvent) => {
      const slot = slotRef.current;
      if (slot && e.target instanceof Node && slot.contains(e.target)) return;
      changeRef.current(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;
      // keyboard users land back on the button, not on a removed pill
      if (menuRef.current?.contains(document.activeElement)) btnRef.current?.focus();
      changeRef.current(false);
    };
    document.addEventListener("pointerdown", onDown, true);
    window.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("pointerdown", onDown, true);
      window.removeEventListener("keydown", onKey);
    };
  }, [open]);

  useEffect(() => { if (!open) { hoverOpenedRef.current = false; cancelLeave(); } }, [open]);
  // the slot can unmount while open (the space frame hides the rail's style
  // button): close, so it does not come back already open
  useEffect(() => () => { cancelLeave(); changeRef.current(false); }, []);

  const isMouse = (e: ReactPointerEvent) => e.pointerType === "mouse";

  return (
    <div
      ref={slotRef}
      className={`vt-rail-menu${props.className ? ` ${props.className}` : ""}${open ? ` vt-rail-menu-open${props.openClassName ? ` ${props.openClassName}` : ""}` : ""}`}
      role="group" aria-label={props.groupLabel}
      data-vt-rail-menu={id}
      onPointerEnter={(e) => { if (isMouse(e)) cancelLeave(); }}
      onPointerLeave={(e) => {
        // hover-dismiss only while the menu sits right beside its button: one
        // that stepped aside is reached across other controls
        if (!isMouse(e) || !open || !besideRef.current) return;
        cancelLeave();
        leaveTimerRef.current = window.setTimeout(() => { leaveTimerRef.current = null; changeRef.current(false); }, LEAVE_GRACE_MS);
      }}
    >
      <button
        ref={btnRef}
        type="button"
        className={props.buttonClassName}
        aria-expanded={open} aria-haspopup="true" aria-controls={open ? menuId : undefined}
        aria-label={props.buttonLabel}
        title={props.buttonTitle}
        data-vt-rail-btn={id}
        onPointerEnter={(e) => {
          if (!isMouse(e) || open) return;
          hoverOpenedRef.current = true;
          onOpenChange(true);
        }}
        onPointerLeave={() => { hoverOpenedRef.current = false; }}
        onClick={() => {
          cancelLeave();
          if (open && hoverOpenedRef.current) { hoverOpenedRef.current = false; return; }
          hoverOpenedRef.current = false;
          onOpenChange(!open);
        }}
      >
        {props.icon}
      </button>
      {open && (
        <div
          ref={menuRef}
          id={menuId}
          className={`vt-rail-flyout${props.menuClassName ? ` ${props.menuClassName}` : ""}`}
          data-vt-rail-flyout={id}
        >
          {props.children}
        </div>
      )}
    </div>
  );
}
