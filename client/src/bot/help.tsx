// Bot page — the "?" beside a term (human directive 2026-10-21: "a small
// question mark by the terminology, the same size as the font, hover over it
// and a window pops up and explains … and it should function even if it's
// detached"). A 1 em glyph; hover, focus or tap opens one small card with the
// glossary line; Esc, blur or leaving closes it. The card is portalled into
// the anchor's OWN document, so it works in docked panels, in-page windows
// and pop-out browser windows alike; it is positioned from the anchor's
// rectangle and clamped to that window, never covering the anchor (Standing
// UI Law). Positioning is pure (placeCard) and pinned by help.test.ts.
import { useCallback, useEffect, useId, useRef, useState, type ReactNode } from "react";
import { createPortal } from "react-dom";
import { GLOSSARY, type GlossaryKey } from "./glossary";

export interface Rect { left: number; top: number; width: number; height: number }
export interface CardPlace { left: number; top: number; above: boolean }
export const CARD_W = 272;
export const CARD_GAP = 6;
export const CARD_MARGIN = 8;

/** Where the card goes: below the anchor, flipped above when the bottom
 *  would leave the window, horizontally clamped inside it. */
export function placeCard(anchor: Rect, cardH: number, win: { w: number; h: number }, cardW = CARD_W): CardPlace {
  const left = Math.max(CARD_MARGIN, Math.min(anchor.left, win.w - cardW - CARD_MARGIN));
  const below = anchor.top + anchor.height + CARD_GAP;
  const fitsBelow = below + cardH <= win.h - CARD_MARGIN;
  const above = !fitsBelow && anchor.top - CARD_GAP - cardH >= CARD_MARGIN;
  const top = above ? anchor.top - CARD_GAP - cardH : fitsBelow ? below : Math.max(CARD_MARGIN, win.h - CARD_MARGIN - cardH);
  return { left, top, above };
}

export function Help({ k, title }: { k: GlossaryKey; title?: ReactNode }) {
  const entry = GLOSSARY[k];
  const btnRef = useRef<HTMLButtonElement>(null);
  const cardRef = useRef<HTMLDivElement>(null);
  const [open, setOpen] = useState(false);
  const [place, setPlace] = useState<CardPlace>({ left: 0, top: 0, above: false });
  const id = useId();
  const closeTimer = useRef<number | null>(null);
  /** what opened the card: a hover or a focus closes on leave / blur, a click
   *  PINS it (leaving no longer closes it; a second click, Esc or a pointer
   *  down outside does) — so a click never toggles away a card the pointer
   *  just opened. */
  const openedBy = useRef<"hover" | "focus" | "click" | null>(null);

  const position = useCallback(() => {
    const b = btnRef.current, c = cardRef.current;
    if (!b || !c) return;
    const view = b.ownerDocument.defaultView;
    if (!view) return;
    const r = b.getBoundingClientRect();
    setPlace(placeCard({ left: r.left, top: r.top, width: r.width, height: r.height }, c.offsetHeight || 80, { w: view.innerWidth, h: view.innerHeight }));
  }, []);

  useEffect(() => {
    if (!open) return;
    position();
    const b = btnRef.current;
    const doc = b?.ownerDocument;
    const view = doc?.defaultView;
    if (!doc || !view) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") { openedBy.current = null; setOpen(false); } };
    const onDown = (e: PointerEvent) => { const t = e.target as Node | null; if (t && !btnRef.current?.contains(t) && !cardRef.current?.contains(t)) { openedBy.current = null; setOpen(false); } };
    doc.addEventListener("keydown", onKey);
    doc.addEventListener("pointerdown", onDown);
    view.addEventListener("resize", position);
    view.addEventListener("scroll", position, true);
    return () => {
      doc.removeEventListener("keydown", onKey);
      doc.removeEventListener("pointerdown", onDown);
      view.removeEventListener("resize", position);
      view.removeEventListener("scroll", position, true);
    };
  }, [open, position]);

  const cancelClose = () => { if (closeTimer.current !== null) { window.clearTimeout(closeTimer.current); closeTimer.current = null; } };
  const hide = () => { openedBy.current = null; setOpen(false); };
  const show = (by: "hover" | "focus" | "click") => { cancelClose(); if (!open || by === "click") openedBy.current = by; setOpen(true); };
  const closeSoon = () => { cancelClose(); if (openedBy.current === "click") return; closeTimer.current = window.setTimeout(hide, 120); };
  useEffect(() => () => cancelClose(), []);

  const doc = btnRef.current?.ownerDocument ?? (typeof document !== "undefined" ? document : null);
  const card = open && doc ? createPortal(
    <div ref={cardRef} id={id} role="tooltip" className="vt-help-card" data-above={place.above ? "true" : undefined} style={{ left: place.left, top: place.top, width: CARD_W }} onPointerEnter={cancelClose} onPointerLeave={closeSoon}>
      <div className="vt-help-term">{title ?? entry.term}</div>
      <div className="vt-help-text">{entry.text}</div>
    </div>,
    doc.body,
  ) : null;

  return (
    <>
      <button
        ref={btnRef}
        type="button"
        className="vt-help"
        aria-label={`What is ${entry.term}?`}
        aria-describedby={open ? id : undefined}
        aria-expanded={open}
        data-help={k}
        onPointerEnter={(e) => { if (e.pointerType === "mouse") show("hover"); }}
        onPointerLeave={(e) => { if (e.pointerType === "mouse") closeSoon(); }}
        onFocus={() => show("focus")}
        onBlur={closeSoon}
        onClick={(e) => { e.stopPropagation(); if (open && openedBy.current === "click") hide(); else show("click"); }}
      >?</button>
      {card}
    </>
  );
}
