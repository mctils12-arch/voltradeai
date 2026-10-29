// Cockpit clock — one re-render per period; the panels' ages and the
// held-since lines read it. 2026-10-29: lifted out of BotPage.tsx, unchanged.
import { useEffect, useState } from "react";

export function useNow(periodMs = 1000): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => { const id = setInterval(() => setNow(Date.now()), periodMs); return () => clearInterval(id); }, [periodMs]);
  return now;
}
