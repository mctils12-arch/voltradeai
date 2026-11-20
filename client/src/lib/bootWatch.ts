// bootWatch — the page half of the boot marker's ownership rule (blackbox.ts
// holds the storage half and explains why, under "WHAT THE MARKER IS A
// STATEMENT ABOUT").
//
// The module that arms the marker is evaluated on every page of the site. This
// watches the PAGE, whatever it is showing:
//   pagehide            leaving any page gracefully is not a crash
//   hidden, no map      a page sent to the background without the map mounted
//                       stands down (a phone discards such a tab with no
//                       pagehide; the marker would accuse the next visit)
//   window over, no map a page that has lived the whole start-up window without
//                       starting the map stands down
// While the map IS mounted nothing here stands down: a death then is exactly
// what the marker exists to catch.
//
// (This listens for the page being HIDDEN. The shared resume signal in
// tabResume.ts is for the opposite edge — coming back — and stays the only way
// a feed poller learns about that.)
//
// Hermetic: the window and document are injected.
import { closeCleanly, standDown, type Storage } from "./blackbox";

export interface BootWatchDeps {
  store: Storage;
  /** true while the map page is mounted */
  mapMounted: () => boolean;
  /** the start-up window, ms */
  windowMs: number;
  win: {
    addEventListener(type: "pagehide", fn: () => void): void;
    setTimeout(fn: () => void, ms: number): unknown;
  };
  doc: {
    readonly visibilityState: string;
    addEventListener(type: "visibilitychange", fn: () => void): void;
  };
}

export function watchBootMarker(d: BootWatchDeps): void {
  d.win.addEventListener("pagehide", () => closeCleanly(d.store));
  const standDownIfNoMap = () => { if (!d.mapMounted()) standDown(d.store); };
  d.doc.addEventListener("visibilitychange", () => { if (d.doc.visibilityState === "hidden") standDownIfNoMap(); });
  d.win.setTimeout(standDownIfNoMap, d.windowMs);
}
