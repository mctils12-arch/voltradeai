// TradeBackdrop — the moving world map behind every Trade screen.
//
// Mounted once by the app shell (home.tsx) while any Trade tab is active, so
// the canvas, its TopoJSON fetch and its particle field survive tab switches
// instead of re-mounting on every visit to Analyze. The animation state comes
// from lib/tradeBackdrop: the Analyze page reports loading / loaded / ticker;
// every other Trade tab gets the idle field (human 2026-10-29: "I want this
// moving map like this but more modern as well for some of the screens").
import { lazy, Suspense, useEffect, useSyncExternalStore } from "react";
import { getTradeBackdrop, subscribeTradeBackdrop } from "@/lib/tradeBackdrop";

// Same stale-chunk recovery the Analyze page carried (a deploy can rename
// the bundle under an open tab): one full reload, then give up to the
// error boundary.
const DataWorldMap = lazy(() =>
  import("@/components/DataWorldMap").catch((err) => {
    const msg = String(err?.message || err);
    const isStaleChunk =
      msg.includes("Failed to fetch dynamically imported module") ||
      msg.includes("Importing a module script failed") ||
      msg.includes("error loading dynamically imported module");
    if (isStaleChunk && !sessionStorage.getItem("__chunk_reloaded")) {
      sessionStorage.setItem("__chunk_reloaded", "1");
      window.location.reload();
      return { default: () => null } as any;
    }
    throw err;
  })
);

export default function TradeBackdrop() {
  const state = useSyncExternalStore(subscribeTradeBackdrop, getTradeBackdrop, getTradeBackdrop);

  // The canvas is position: fixed behind the page; the body has to be
  // transparent for it to show through on phones (moved here from
  // analyze.tsx so it holds for every Trade tab, and is undone on leave).
  useEffect(() => {
    const prev = document.body.style.background;
    document.body.style.background = "transparent";
    return () => { document.body.style.background = prev; };
  }, []);

  return (
    <Suspense fallback={null}>
      <DataWorldMap isLoading={state.isLoading} hasData={state.hasData} ticker={state.ticker} />
    </Suspense>
  );
}
