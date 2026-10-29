// Cockpit viewport — which tier this window falls in, its size, and whether
// the pointer is coarse (touch: no hover, no pop-outs); plus the
// reduced-motion query the FLIP honours. 2026-10-29: lifted out of
// BotPage.tsx, unchanged.
import { useEffect, useState } from "react";
import { tierFor, type Tier } from "./zoneModel";

export interface Viewport { tier: Tier; w: number; h: number; touch: boolean }

export function useTier(): Viewport {
  const read = () => ({ w: typeof window === "undefined" ? 1440 : window.innerWidth, h: typeof window === "undefined" ? 900 : window.innerHeight });
  const [vp, setVp] = useState(read);
  useEffect(() => { const on = () => setVp(read()); window.addEventListener("resize", on); window.addEventListener("orientationchange", on); return () => { window.removeEventListener("resize", on); window.removeEventListener("orientationchange", on); }; }, []);
  const touch = typeof window !== "undefined" && typeof window.matchMedia === "function" ? window.matchMedia("(pointer: coarse)").matches : false;
  return { tier: tierFor(vp.w), w: vp.w, h: vp.h, touch };
}

export function reducedMotion(): boolean {
  return typeof window !== "undefined" && typeof window.matchMedia === "function" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}
