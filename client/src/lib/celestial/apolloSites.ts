// LUNAR SURFACE MISSIONS — the persisted panel TOGGLE only.
//
// The site DATA that used to live here (APOLLO_SITES, the imagery note, the
// near-side note, lrocFeaturedUrl) moved to lunarMissions.ts on 2026-08-13,
// when the six Apollo landings became 35 verified sites across every agency
// that has reached the Moon. Two arrays keyed on the same ids would drift, so
// there is exactly one; and the old near-side note ("none are on the far
// side") became factually false the moment Chang'e 4/6 and LADEE landed in
// the data — it is now COMPUTED from the array in lunarMissions.ts.
//
// This file keeps only the preference store, because its localStorage KEY
// must not change: renaming `vt.celestial.apolloSites` would silently
// re-enable the layer for every user who had turned it off.

// ── persisted preference (the orbitPath/scaleModel localStorage pattern) ─────
// Human directive 2026-08-12: "i want it shown on the moon and a toggle in
// the layers under celestial". The markers shipped in v1.0.668 with no way
// to turn them off; this is that switch, and it lives with its data the way
// orbitPath.ts hosts its own.
//
// DEFAULT ON. These are documented survey coordinates drawn on a real
// imagery mosaic — RAW geometry with a stated source, not a prediction, so
// no ladder gate applies (same reasoning as orbit paths).

export const APOLLO_SITES_PREF_KEY = "vt.celestial.apolloSites";

export function readApolloSitesPref(): boolean {
  try {
    const raw = globalThis.localStorage?.getItem(APOLLO_SITES_PREF_KEY);
    if (raw === "0") return false;
    if (raw === "1") return true;
  } catch { /* no storage — default */ }
  return true;
}

let apolloOn: boolean = readApolloSitesPref();
const apolloListeners = new Set<() => void>();

export function getApolloSitesPref(): boolean {
  return apolloOn;
}

export function setApolloSitesPref(on: boolean): void {
  if (on === apolloOn) return;
  apolloOn = on;
  try {
    globalThis.localStorage?.setItem(APOLLO_SITES_PREF_KEY, on ? "1" : "0");
  } catch { /* best-effort */ }
  for (const fn of Array.from(apolloListeners)) {
    try { fn(); } catch { /* listener owns its errors */ }
  }
}

export function subscribeApolloSitesPref(fn: () => void): () => void {
  apolloListeners.add(fn);
  return () => { apolloListeners.delete(fn); };
}
