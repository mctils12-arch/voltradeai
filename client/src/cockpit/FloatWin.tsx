// Cockpit float window — a panel opened as a window over the page (its ghost
// slot kept). Drag the title bar to move, the corner to resize; the rect is
// clamped to the page and snaps to its edges (zoneModel.ts). 2026-10-29:
// lifted out of BotPage.tsx for the shared engine; the page passes the title
// (the panel id only ever named it) — markup and classes unchanged.
import { useRef, type ReactNode } from "react";
import { clampRect, snapEdges, type Rect } from "./zoneModel";
import { ICONS } from "./icons";

export interface FloatWinProps<Id extends string = string> {
  /** the panel's id, written as data-win for a page's harness selectors (the bot passes none: its markup predates it) */
  id?: Id;
  title: string; rect: Rect; page: { w: number; h: number };
  onRect: (r: Rect) => void; onDock: () => void; onZoom: () => void; onPop: () => void;
  /** the "own window" control is offered (default true; a page with no pane route passes false) */
  canPop?: boolean;
  children: ReactNode;
}

export function FloatWin<Id extends string = string>({ id, title, rect, page, onRect, onDock, onZoom, onPop, canPop = true, children }: FloatWinProps<Id>) {
  const drag = useRef<{ mode: "move" | "size"; sx: number; sy: number; start: Rect } | null>(null);
  // A press on a title-bar BUTTON is not a drag: capturing the pointer to the
  // bar sent the button's click to the bar instead, so dock / zoom / close
  // never fired (field report 2026-10-20; bot-ui harness battery).
  const onDown = (mode: "move" | "size") => (e: React.PointerEvent) => {
    if (e.button !== 0 || (e.target as HTMLElement).closest("button")) return;
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    drag.current = { mode, sx: e.clientX, sy: e.clientY, start: rect };
  };
  const onMove = (e: React.PointerEvent) => {
    const g = drag.current; if (!g) return;
    const dx = e.clientX - g.sx, dy = e.clientY - g.sy;
    const next = g.mode === "move" ? { ...g.start, x: g.start.x + dx, y: g.start.y + dy } : { ...g.start, w: g.start.w + dx, h: g.start.h + dy };
    onRect(snapEdges(clampRect(next, page), page));
  };
  const onUp = () => { drag.current = null; };
  return (
    <div className="vt-bot-win" role="dialog" aria-label={`${title} window`} data-win={id} style={{ left: rect.x, top: rect.y, width: rect.w, height: rect.h }}>
      <div className="vt-bot-wh" onPointerDown={onDown("move")} onPointerMove={onMove} onPointerUp={onUp} onPointerCancel={onUp}>
        <span className="vt-bot-ti">{title}</span>
        <span className="vt-bot-t3 vt-bot-m" style={{ fontSize: 11 }}>floating · ghost slot kept · Esc docks</span>
        <button type="button" className="vt-bot-wbtn" onClick={onDock}>{ICONS.dock} dock</button>
        <button type="button" className="vt-bot-wbtn" onClick={onZoom}>{ICONS.zoom} zoom</button>
        {canPop ? <button type="button" className="vt-bot-wbtn" onClick={onPop} title="Its own browser window — drag it to another monitor">{ICONS.pop} own window</button> : null}
        <button type="button" className="vt-bot-wbtn" aria-label="Close window" onClick={onDock}>{ICONS.close}</button>
      </div>
      <div className="vt-bot-wb">{children}</div>
      <div className="vt-bot-resize" aria-hidden="true" onPointerDown={onDown("size")} onPointerMove={onMove} onPointerUp={onUp} onPointerCancel={onUp} />
    </div>
  );
}
