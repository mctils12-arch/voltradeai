/** Where a FLIP slot sits in its scroll host's CONTENT coordinates. Viewport
 *  rects move when the host scrolls even though no slot moved, so a float
 *  toggle that follows a scroll would otherwise spring every slot by the
 *  scroll delta. Content coordinates are scroll-invariant.
 *  2026-10-29: moved from client/src/bot/flipGeometry.ts with useFlip (the
 *  shared cockpit engine); the bot path re-exports it. */
export interface Box { left: number; top: number }
export interface Host extends Box { scrollLeft: number; scrollTop: number }

export function contentPos(rect: Box, host: Host | null): Box {
  if (!host) return { left: rect.left, top: rect.top };
  return { left: rect.left - host.left + host.scrollLeft, top: rect.top - host.top + host.scrollTop };
}
