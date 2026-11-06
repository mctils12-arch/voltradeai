// Cockpit FLIP — layout motion on frameCore springs (Rendering & Motion Law I:
// one rAF loop; transforms only during motion). When the layout key changes,
// every slot that moved springs from its old box to its new one with a
// transform; on settle the transform is dropped and CSS grid owns the final
// box. Slots are measured in their scroll host's CONTENT coordinates
// (flipGeometry.ts) so a scroll is not a move. 2026-10-29: lifted out of
// BotPage.tsx for the shared cockpit engine, generic over the panel id; the
// host selector and the HUD label are the only knobs.
import { useEffect, useLayoutEffect, useMemo, useRef, type MutableRefObject } from "react";
import { frameCore, PRIORITY, type Spring } from "@/render/frameCore";
import { contentPos, type Box } from "./flipGeometry";
import { reducedMotion } from "./useTier";

export interface FlipOptions {
  /** the scroll host the slots are measured against (the cockpit grid) */
  host?: string;
  /** the registration's label in the ?perf=1 HUD */
  label?: string;
}

/** the slot springs: critically damped (26 ≈ 2·√170) — fast, no overshoot, no ring */
const FLIP_SPRING = { stiffness: 170, damping: 26 };

export function useFlip<Id extends string>(layoutKey: string, slots: MutableRefObject<Map<Id, HTMLElement>>, opts: FlipOptions = {}): void {
  const host = opts.host ?? ".vt-bot-grid";
  const label = opts.label ?? "cockpit-flip";
  const prev = useRef<Map<Id, Box>>(new Map());
  const anim = useRef<Map<Id, { x: Spring; y: Spring; detach: () => void }>>(new Map());
  const loop = useMemo(() => frameCore(), []);
  useEffect(() => {
    loop.start();
    const un = loop.register(() => {
      for (const [id, a] of anim.current) {
        const el = slots.current.get(id);
        if (!el) continue;
        if (a.x.asleep && a.y.asleep) { el.style.transform = ""; a.detach(); anim.current.delete(id); continue; }
        el.style.transform = `translate(${a.x.value.toFixed(2)}px, ${a.y.value.toFixed(2)}px)`;
      }
    }, PRIORITY.RENDER, { label });
    return () => { un(); for (const a of anim.current.values()) a.detach(); anim.current.clear(); };
  }, [loop, slots, label]);
  useLayoutEffect(() => {
    const next = new Map<Id, Box>();
    for (const [id, el] of slots.current) {
      const g = el.closest<HTMLElement>(host);
      const gr = g?.getBoundingClientRect();
      next.set(id, contentPos(el.getBoundingClientRect(), g && gr ? { left: gr.left, top: gr.top, scrollLeft: g.scrollLeft, scrollTop: g.scrollTop } : null));
    }
    if (!reducedMotion()) {
      for (const [id, rect] of next) {
        const was = prev.current.get(id);
        if (!was) continue;
        const dx = was.left - rect.left, dy = was.top - rect.top;
        if (Math.abs(dx) < 1 && Math.abs(dy) < 1) continue;
        const cur = anim.current.get(id);
        if (cur) { cur.x.snap(dx + cur.x.value); cur.y.snap(dy + cur.y.value); cur.x.setTarget(0); cur.y.setTarget(0); continue; }
        const sx = loop.createSpring(dx, FLIP_SPRING); const sy = loop.createSpring(dy, FLIP_SPRING);
        sx.spring.setTarget(0); sy.spring.setTarget(0);
        anim.current.set(id, { x: sx.spring, y: sy.spring, detach: () => { sx.detach(); sy.detach(); } });
      }
    }
    prev.current = next;
  }, [layoutKey, loop, slots, host]);
}
